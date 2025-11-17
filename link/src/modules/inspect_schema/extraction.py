"""Step 3 of LinkAlign: select relevant tables/columns."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .prompts import build_auditor_messages
from .types import ColumnMetadata, DatabaseMetadata, ExtractionResult, TableMetadata


# DESIGN NOTE:
# In 'llm' mode, we use a Hugging Face text-generation pipeline as a
# "column auditor" (similar to LinkAlign). For each table:
#   1) Build a chat-style prompt with:
#       - system: instructions to return ONLY JSON of the form:
#           { "columns": [ { "name": ..., "needed": true/false, "reason": ... }, ... ] }
#       - user: includes the question and the table schema (columns + types).
#   2) Call the pipeline with messages and decode the generated text.
#   3) Parse the JSON, align it with the existing ColumnMetadata entries,
#      and set col.extra["needed"] and col.extra["reason"] accordingly.
#   4) Keep only tables where at least one column is needed.
#   5) Compute coverage_stats (num_columns_total, num_columns_needed, fraction, etc.).
#
# The HF pipeline instance is passed via config["llm_pipeline"].
# If the JSON parsing fails, we fall back to marking all columns as not needed.


def extract_relevant_items(
    question: str,
    filtered_dbs: List[DatabaseMetadata],
    config: Dict[str, Any],
) -> ExtractionResult:
    """
    Produce the schema subset that matters for the supplied question.
    
    Supports two modes:
    - "heuristic": keyword matching (default, fast)
    - "llm": uses HuggingFace model to decide column relevance
    """
    if not filtered_dbs:
        # Return empty result if no databases
        empty_db = DatabaseMetadata(db_id="none", engine="none", tables=[])
        return ExtractionResult(
            selected_schema=empty_db,
            coverage_stats={"num_tables": 0, "num_columns_total": 0, "num_columns_needed": 0},
            debug_info={"reason": "No filtered databases provided"}
        )

    # Check extraction mode
    extraction_config = config.get("extraction", {})
    mode = extraction_config.get("mode", "heuristic")
    
    if mode == "llm":
        return _extract_with_llm(question, filtered_dbs, config)
    else:
        return _extract_with_heuristics(question, filtered_dbs, config)


def _extract_with_heuristics(
    question: str,
    filtered_dbs: List[DatabaseMetadata],
    config: Dict[str, Any],
) -> ExtractionResult:
    """Original heuristic-based extraction using keyword matching."""
    db = filtered_dbs[0]
    question_lower = question.lower()

    selected_tables = []
    total_columns = 0
    needed_columns = 0

    for table in db.tables:
        table_columns = []
        table_has_needed = False

        for col in table.columns:
            total_columns += 1
            col_name_lower = col.name.lower()

            # Heuristic: column is needed if:
            # 1. Its name appears in the question, OR
            # 2. Question asks "how many" and column is "number", OR
            # 3. Question asks "count" and column is "number"
            is_needed = (
                col_name_lower in question_lower or
                ("how many" in question_lower and col_name_lower == "number") or
                ("count" in question_lower and col_name_lower == "number")
            )

            if is_needed:
                needed_columns += 1
                table_has_needed = True

            reason = "Column name found in question" if is_needed else "Not explicitly mentioned"
            table_columns.append(
                ColumnMetadata(
                    name=col.name,
                    type=col.type,
                    description=col.description,
                    extra={**col.extra, "needed": is_needed, "reason": reason}
                )
            )

        if table_has_needed:
            selected_tables.append(
                TableMetadata(
                    table_name=table.table_name,
                    full_name=table.full_name,
                    columns=table_columns,
                    description=table.description,
                    extra={**table.extra, "reason": "Contains columns relevant to the question"}
                )
            )

    selected_schema = DatabaseMetadata(
        db_id=db.db_id,
        engine=db.engine,
        tables=selected_tables,
        extra=db.extra
    )

    return ExtractionResult(
        selected_schema=selected_schema,
        coverage_stats={
            "num_tables": len(selected_tables),
            "num_columns_total": total_columns,
            "num_columns_needed": needed_columns,
            "fraction_columns_needed": needed_columns / total_columns if total_columns > 0 else 0.0
        },
        debug_info={"question": question, "db_id": db.db_id, "mode": "heuristic"}
    )


def _extract_with_llm(
    question: str,
    filtered_dbs: List[DatabaseMetadata],
    config: Dict[str, Any],
) -> ExtractionResult:
    """
    Use LLM (HF pipeline) to decide which columns are needed,
    following the LinkAlign column auditor pattern.
    """
    db = filtered_dbs[0]  # Process 1: single database
    
    # Get LLM pipeline from config
    llm_pipeline = config.get("llm_pipeline")
    if llm_pipeline is None:
        raise ValueError(
            "LLM extraction mode requires 'llm_pipeline' in config. "
            "Pass a transformers.pipeline instance or switch to mode='heuristic'."
        )
    extraction_config = config.get("extraction", {})
    max_new_tokens = int(extraction_config.get("llm_max_new_tokens", 256))
    temperature = float(extraction_config.get("llm_temperature", 0.0))

    print(f"[DEBUG] LLM max_new_tokens={max_new_tokens}, temperature={temperature}")

    selected_tables: List[TableMetadata] = []
    total_columns = 0
    needed_columns = 0

    for table in db.tables:
        # 1) Build messages for this table
        messages = build_auditor_messages(question, table, extraction_config)
        
        # 2) Call the LLM
        try:
            # Get tokenizer if available
            tokenizer = config.get("llm_tokenizer")
            
            # If tokenizer is available, format with chat template
            if tokenizer:
                # Apply chat template and tokenize=False to get text
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
                # Call pipeline with formatted text
                raw_output = llm_pipeline(
                    formatted_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=False,  # con temperature=0.0, fijo
                    repetition_penalty=extraction_config.get("llm_repetition_penalty", 1.0),
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_full_text=False, 
                )
            else:
                # Fallback: send messages directly (may not work well)
                raw_output = llm_pipeline(
                    messages,
                    max_new_tokens=extraction_config.get("llm_max_new_tokens", 256),
                    temperature=extraction_config.get("llm_temperature", 0.2),
                    do_sample=extraction_config.get("llm_temperature", 0.2) > 0.0,
                )
            
            # 3) Parse JSON from output
            json_obj = _parse_auditor_json(raw_output)
            print(f"[LLM] Auditing table {table.full_name}...")
            print(f"[LLM] Raw output: {raw_output}")
            print(f"[LLM] Parsed JSON: {json.dumps(json_obj, indent=2, default=str)}")

            
        except Exception as e:
            # Fallback: mark all columns as not needed
            json_obj = {"columns": []}
            print(f"Warning: LLM extraction failed for table {table.table_name}: {e}")
        
        # 4) Map columns to ColumnMetadata with extra["needed"]
        table_columns: List[ColumnMetadata] = []
        table_has_needed = False
        col_by_name = {c.name: c for c in table.columns}
        
        for col_name, col_meta in col_by_name.items():
            total_columns += 1
            col_info = _lookup_col_info(json_obj, col_name)
            
            if col_info is not None:
                is_needed = bool(col_info.get("needed", False))
                reason = col_info.get("reason", "Decided by LLM auditor")
            else:
                # Default: not needed if LLM didn't mention it
                is_needed = False
                reason = "Not selected by LLM auditor"
            
            if is_needed:
                needed_columns += 1
                table_has_needed = True
            
            table_columns.append(
                ColumnMetadata(
                    name=col_meta.name,
                    type=col_meta.type,
                    description=col_meta.description,
                    extra={**col_meta.extra, "needed": is_needed, "reason": reason}
                )
            )
        
        # 5) Include table only if it has at least one needed column
        if table_has_needed:
            selected_tables.append(
                TableMetadata(
                    table_name=table.table_name,
                    full_name=table.full_name,
                    columns=table_columns,
                    description=table.description,
                    extra={**table.extra, "reason": "Selected by LLM auditor"}
                )
            )
    
    # 6) Build selected schema and stats
    selected_schema = DatabaseMetadata(
        db_id=db.db_id,
        engine=db.engine,
        tables=selected_tables,
        extra=db.extra
    )
    
    coverage_stats = {
        "num_tables": len(selected_tables),
        "num_columns_total": total_columns,
        "num_columns_needed": needed_columns,
        "fraction_columns_needed": needed_columns / total_columns if total_columns > 0 else 0.0
    }
    
    return ExtractionResult(
        selected_schema=selected_schema,
        coverage_stats=coverage_stats,
        debug_info={"question": question, "db_id": db.db_id, "mode": "llm"}
    )


def _parse_auditor_json(raw_output: Any) -> Dict[str, Any]:
    """
    Extract and parse JSON from the LLM's raw output.

    Primero intenta usar el generated_text tal cual.
    Si falla, intenta recortar desde la primera '{' y balancear llaves.
    Si aun así falla, devuelve {"columns": []}.
    """
    # 1) Extraer texto base
    if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], dict):
        gen_text = raw_output[0].get("generated_text", "")
        if isinstance(gen_text, list):
            # Chat-style: nos quedamos con el último mensaje
            gen_text = gen_text[-1].get("content", "") if gen_text else ""
    elif isinstance(raw_output, dict):
        gen_text = raw_output.get("generated_text", str(raw_output))
    else:
        gen_text = str(raw_output)

    # 2) Limpieza ligera (por si algún modelo mete tokens raros)
    assistant_markers = [
        "<|start_header_id|>assistant<|end_header_id|>",
        "assistant\n",
        "[/INST]",
    ]
    for marker in assistant_markers:
        if marker in gen_text:
            gen_text = gen_text.split(marker)[-1]

    for token in ["<|eot_id|>", "</s>", "<|endoftext|>"]:
        gen_text = gen_text.replace(token, "")

    fence_markers = ["```json", "```JSON", "```", "~~~~"]
    for marker in fence_markers:
        if marker in gen_text:
            gen_text = gen_text.split(marker)[-1]
    gen_text = gen_text.strip()

    # === PRIMER INTENTO: usar el texto tal cual ===
    try:
        parsed = json.loads(gen_text)
        if "columns" in parsed and isinstance(parsed["columns"], list):
            return parsed
        else:
            print(f"[WARNING] Parsed JSON missing 'columns' field: {parsed}")
            return {"columns": []}
    except json.JSONDecodeError:
        # Seguimos al segundo intento
        pass

    # === SEGUNDO INTENTO: recortar desde la primera '{' y balancear ===
    start_idx = gen_text.find("{")
    if start_idx == -1:
        print(f"[WARNING] No JSON object found in LLM output: {gen_text[:200]}...")
        return {"columns": []}

    json_str = gen_text[start_idx:]
    json_str = _balance_json_braces(json_str)

    try:
        parsed = json.loads(json_str)
        if "columns" in parsed and isinstance(parsed["columns"], list):
            return parsed
        else:
            print(f"[WARNING] Parsed JSON (balanced) missing 'columns' field: {parsed}")
            return {"columns": []}
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON decode failed after balancing: {e}")
        print(f"[ERROR] Attempted to parse: {json_str[:300]}")
        return {"columns": []}

def _balance_json_braces(json_str: str) -> str:
    """Append missing closing braces to balance a JSON string."""
    open_braces = 0
    for char in json_str:
        if char == "{":
            open_braces += 1
        elif char == "}":
            open_braces = max(open_braces - 1, 0)
    return json_str + ("}" * open_braces)


def _lookup_col_info(json_obj: Dict[str, Any], col_name: str) -> Optional[Dict[str, Any]]:
    """
    Look up a column's info (needed, reason) from the parsed JSON.
    
    Returns None if the column is not found.
    """
    columns = json_obj.get("columns", [])
    for col in columns:
        if col.get("name") == col_name:
            return col
    return None
