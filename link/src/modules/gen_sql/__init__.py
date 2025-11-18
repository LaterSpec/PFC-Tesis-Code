"""Orchestrator for the gen_sql generation pipeline."""
from __future__ import annotations

from typing import Any, Dict

from . import generator, signals, validation
from .types import GenerationSignals


def gen_sql(
    question: str,
    schema_context: Dict[str, Any],
    gen_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate SQL query from natural language question using LLM.
    
    This is the main entry point for SQL generation, following the same
    pattern as inspect_schema module.
    
    Args:
        question: Natural language question
        schema_context: Output from inspect_schema containing:
            - schema_text: str
            - selected_schema: DatabaseMetadata
            - tokens_estimate: int
        gen_config: Configuration dict containing:
            - llm_pipeline: HuggingFace pipeline instance
            - llm_tokenizer: HuggingFace tokenizer instance
            - generation: dict with HF generation params
            - num_candidates: int (default 1)
            
    Returns:
        Dictionary with:
            - sql: str (primary candidate)
            - expected_shape: dict
            - candidates: list of all generated candidates
            - gen_signals: dict of numerical signals for SCM
            - debug_info: dict with generation details
    """
    # Step 1: Extract engine from schema context
    selected_schema = schema_context.get("selected_schema")
    engine = selected_schema.engine if selected_schema else "snowflake"
    
    # Step 2: Generate SQL candidate(s) using LLM
    generation_result = generator.generate_sql_candidates(
        question=question,
        schema_context=schema_context,
        gen_config=gen_config
    )
    
    # Step 3: Validate candidates (static checks, no execution)
    validated_candidates, aggregate_info = validation.validate_candidates(
        candidates=generation_result.candidates,
        engine=engine
    )
    
    # Update generation result with validated candidates
    generation_result.candidates = validated_candidates
    if validated_candidates:
        generation_result.primary = validated_candidates[0]
    
    # Step 4: Build signals for SCM
    generation_signals = signals.build_generation_signals(
        result=generation_result,
        aggregate_info=aggregate_info
    )
    
    # Step 5: Package output
    return {
        "sql": generation_result.primary.sql,
        "expected_shape": generation_result.primary.expected_shape,
        "candidates": [
            {
                "sql": cand.sql,
                "expected_shape": cand.expected_shape,
                "rationale": cand.rationale,
                "extra": cand.extra,
            }
            for cand in generation_result.candidates
        ],
        "gen_signals": generation_signals.values,
        "debug_info": {
            "question": question,
            "engine": engine,
            "format_ok": generation_result.format_ok,
            "validation_errors": generation_result.validation_errors,
            "aggregate_info": aggregate_info,
        }
    }
