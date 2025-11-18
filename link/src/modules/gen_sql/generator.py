"""SQL generation using LLM and JSON parsing."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from .prompts import build_generator_messages
from .types import GenerationResult, SqlCandidate


def generate_sql_candidates(
    question: str,
    schema_context: Dict[str, Any],
    gen_config: Dict[str, Any]
) -> GenerationResult:
    """
    Generate SQL candidate(s) using LLM.
    
    Supports generating multiple candidates if num_candidates > 1.
    Uses HuggingFace pipeline with chat template formatting.
    
    Args:
        question: Natural language question
        schema_context: Output from inspect_schema module
        gen_config: Configuration including llm_pipeline, llm_tokenizer, etc.
        
    Returns:
        GenerationResult with candidates and validation status
    """
    llm_pipeline = gen_config.get("llm_pipeline")
    if llm_pipeline is None:
        raise ValueError(
            "gen_sql requires 'llm_pipeline' in gen_config. "
            "Pass a transformers.pipeline instance."
        )
    
    tokenizer = gen_config.get("llm_tokenizer")
    generation_params = gen_config.get("generation", {})
    num_candidates = gen_config.get("num_candidates", 1)
    
    # Build messages for LLM
    messages = build_generator_messages(question, schema_context, gen_config)
    
    candidates: List[SqlCandidate] = []
    debug_outputs: List[Any] = []
    
    # Generate candidates (currently one at a time; could batch if num_return_sequences supported)
    for i in range(num_candidates):
        try:
            # Call LLM
            raw_output = _call_llm(
                messages=messages,
                llm_pipeline=llm_pipeline,
                tokenizer=tokenizer,
                generation_params=generation_params
            )
            
            debug_outputs.append(raw_output)
            
            # Parse JSON response
            parsed = _parse_generator_json(raw_output)
            
            # Create candidate
            candidate = SqlCandidate(
                sql=parsed.get("sql", ""),
                expected_shape=parsed.get("expected_shape", {}),
                rationale=parsed.get("rationale"),
                extra={
                    "raw_output": raw_output,
                    "candidate_index": i,
                }
            )
            
            candidates.append(candidate)
            
            print(f"[GEN_SQL] Generated candidate {i+1}/{num_candidates}")
            print(f"[GEN_SQL] SQL: {candidate.sql[:100]}...")
            
        except Exception as e:
            print(f"[ERROR] Failed to generate candidate {i+1}: {e}")
            # Add empty candidate on failure
            candidates.append(
                SqlCandidate(
                    sql="",
                    expected_shape={},
                    rationale=None,
                    extra={"error": str(e), "candidate_index": i}
                )
            )
    
    # Select primary candidate (first one for now; could add ranking later)
    primary = candidates[0] if candidates else SqlCandidate(sql="", expected_shape={})
    
    # Basic format check
    format_ok = bool(primary.sql.strip())
    validation_errors = []
    if not format_ok:
        validation_errors.append("Primary candidate has empty SQL")
    
    return GenerationResult(
        primary=primary,
        candidates=candidates,
        format_ok=format_ok,
        validation_errors=validation_errors,
        debug_info={
            "question": question,
            "messages": messages,
            "raw_outputs": debug_outputs,
        }
    )


def _call_llm(
    messages: List[Dict[str, str]],
    llm_pipeline: Any,
    tokenizer: Any,
    generation_params: Dict[str, Any]
) -> Any:
    """
    Call HuggingFace pipeline with chat-formatted messages.
    
    Similar to extraction._extract_with_llm pattern.
    
    Args:
        messages: Chat messages (system + user)
        llm_pipeline: HuggingFace pipeline instance
        tokenizer: HuggingFace tokenizer (for chat template)
        generation_params: Generation parameters (max_new_tokens, temperature, etc.)
        
    Returns:
        Raw output from pipeline
    """
    # Extract generation parameters
    max_new_tokens = generation_params.get("max_new_tokens", 512)
    temperature = generation_params.get("temperature", 0.0)
    do_sample = generation_params.get("do_sample", False)
    top_p = generation_params.get("top_p", 1.0)
    repetition_penalty = generation_params.get("repetition_penalty", 1.0)
    
    print(f"[DEBUG] LLM params: max_new_tokens={max_new_tokens}, temperature={temperature}")
    
    # Format with chat template if tokenizer available
    if tokenizer:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        raw_output = llm_pipeline(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )
    else:
        # Fallback: send messages directly (may not work well)
        raw_output = llm_pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
    
    return raw_output


def _parse_generator_json(raw_output: Any) -> Dict[str, Any]:
    """
    Extract and parse JSON from LLM's raw output.
    
    Handles different output formats and applies robust cleaning/balancing
    similar to extraction._parse_auditor_json.
    
    Args:
        raw_output: Raw output from HuggingFace pipeline
        
    Returns:
        Parsed dict with keys: sql, expected_shape, rationale
    """
    # Extract text from pipeline output
    if isinstance(raw_output, list) and len(raw_output) > 0:
        if isinstance(raw_output[0], dict):
            gen_text = raw_output[0].get("generated_text", "")
            if isinstance(gen_text, list):
                # Chat format: last message contains response
                gen_text = gen_text[-1].get("content", "") if gen_text else ""
        else:
            gen_text = str(raw_output[0])
    elif isinstance(raw_output, dict):
        gen_text = raw_output.get("generated_text", str(raw_output))
    else:
        gen_text = str(raw_output)
    
    # Extract assistant's response (for Llama-3.2, Mistral, etc.)
    assistant_markers = [
        "<|start_header_id|>assistant<|end_header_id|>",
        "assistant\n",
        "[/INST]",
    ]
    
    for marker in assistant_markers:
        if marker in gen_text:
            parts = gen_text.split(marker)
            if len(parts) > 1:
                gen_text = parts[-1]
                break
    
    # Remove end-of-text tokens
    eot_tokens = ["<|eot_id|>", "</s>", "<|endoftext|>"]
    for token in eot_tokens:
        gen_text = gen_text.replace(token, "")
    
    # Strip markdown fences
    fence_markers = ["```json", "```JSON", "```sql", "```SQL", "```", "~~~~"]
    for marker in fence_markers:
        if marker in gen_text:
            gen_text = gen_text.split(marker)[-1]
    gen_text = gen_text.replace("```", "").replace("~~~~", "").strip()
    
    # Find JSON object
    try:
        start_idx = gen_text.find("{")
        end_idx = gen_text.rfind("}")
        
        if start_idx != -1:
            if end_idx == -1 or end_idx <= start_idx:
                # Balance braces if truncated
                json_str = gen_text[start_idx:]
                json_str = _balance_json_braces(json_str)
            else:
                json_str = gen_text[start_idx:end_idx + 1]
            
            # First attempt: try parsing as-is
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError as e:
                # Second attempt: try to fix common issues
                print(f"[DEBUG] Initial parse failed, attempting repair: {e}")
                
                # Add missing rationale field if needed
                if '"expected_shape"' in json_str and '"rationale"' not in json_str:
                    # Find the closing of expected_shape
                    exp_shape_start = json_str.find('"expected_shape"')
                    if exp_shape_start != -1:
                        # Find the closing } of the expected_shape object
                        brace_count = 0
                        found_start = False
                        closing_idx = -1
                        
                        for i in range(exp_shape_start, len(json_str)):
                            if json_str[i] == '{':
                                brace_count += 1
                                found_start = True
                            elif json_str[i] == '}':
                                brace_count -= 1
                                if found_start and brace_count == 0:
                                    closing_idx = i + 1
                                    break
                        
                        if closing_idx != -1:
                            # Insert comma and rationale if not present
                            rest = json_str[closing_idx:].strip()
                            if not rest.startswith(','):
                                json_str = json_str[:closing_idx] + ',\n  "rationale": ""' + json_str[closing_idx:]
                
                # Balance braces again after modifications
                json_str = _balance_json_braces(json_str)
                
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError as e2:
                    print(f"[ERROR] Repair attempt failed: {e2}")
                    print(f"[ERROR] Final JSON: {json_str[:300]}")
                    return {"sql": "", "expected_shape": {}, "rationale": ""}
            
            # Validate expected structure
            if "sql" in parsed:
                return {
                    "sql": parsed.get("sql", ""),
                    "expected_shape": parsed.get("expected_shape", {}),
                    "rationale": parsed.get("rationale", "")
                }
            else:
                print(f"[WARNING] Parsed JSON missing 'sql' field: {parsed}")
                return {"sql": "", "expected_shape": {}, "rationale": ""}
        else:
            print(f"[WARNING] No JSON found in: {gen_text[:200]}...")
            return {"sql": "", "expected_shape": {}, "rationale": ""}
    
    except Exception as e:
        print(f"[ERROR] Unexpected error during JSON parsing: {e}")
        print(f"[ERROR] Context: {gen_text[:300]}")
        return {"sql": "", "expected_shape": {}, "rationale": ""}


def _balance_json_braces(json_str: str) -> str:
    """
    Append missing closing braces/brackets to balance a JSON string.
    
    Handles both curly braces {} and square brackets [].
    """
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape_next = False
    
    for char in json_str:
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if not in_string:
            if char == "{":
                open_braces += 1
            elif char == "}":
                open_braces = max(open_braces - 1, 0)
            elif char == "[":
                open_brackets += 1
            elif char == "]":
                open_brackets = max(open_brackets - 1, 0)
    
    # Close any unclosed structures
    result = json_str
    result += ("]" * open_brackets)
    result += ("}" * open_braces)
    
    return result
