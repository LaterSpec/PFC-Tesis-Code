"""Orchestrator for the inspect_schema perception pipeline."""
from __future__ import annotations

from typing import Any, Dict

from . import extraction, filtering, metadata_loader, retrieval, schema_index, serialization, signals
from .types import SchemaContext, SchemaSignals


def inspect_schema(question: str, db_config: Dict[str, Any], inspect_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the end-to-end inspect_schema pipeline and return structured outputs."""
    # Step 1: Load database metadata (from BigQuery or cache)
    databases = metadata_loader.load_db_metadata(db_config)

    # Step 2: (Future) Build schema index for multi-DB retrieval
    # For now, we pass an empty index
    index = {}

    # Step 3: Initial retrieval of candidate databases (Step 1 LinkAlign)
    max_candidates = inspect_config.get("max_candidates", 5)
    retrieval_result = retrieval.initial_retrieval(
        question=question,
        databases=databases,
        index=index,
        max_candidates=max_candidates
    )

    # Step 4: (Optional) Query rewriting and scoring
    rewrite_info = retrieval.rewrite_and_score_queries(
        question=question,
        retrieval_result=retrieval_result,
        databases=databases,
        config=inspect_config
    )

    # Step 5: Filter irrelevant schemas (Step 2 LinkAlign)
    max_final_schemas = inspect_config.get("max_final_schemas", 1)
    filter_result = filtering.filter_irrelevant_schemas(
        question=question,
        retrieval_result=retrieval_result,
        databases=databases,
        rewrite_info=rewrite_info,
        max_final_schemas=max_final_schemas
    )

    # Step 6: Extract relevant tables and columns (Step 3 LinkAlign)
    # Build list of DatabaseMetadata for filtered candidates
    db_map = {db.db_id: db for db in databases}
    filtered_db_list = [
        db_map[cand["db_id"]]
        for cand in filter_result.filtered_candidates
        if cand["db_id"] in db_map
    ]

    extraction_result = extraction.extract_relevant_items(
        question=question,
        filtered_dbs=filtered_db_list,
        config=inspect_config
    )

    # Step 7: Serialize schema to text for LLM
    serialization_config = inspect_config.get("serialization", {})
    schema_context = serialization.serialize_schema_context(
        selected_schema=extraction_result.selected_schema,
        serialization_config=serialization_config
    )

    # Step 8: Build schema signals for SCM
    schema_signals = signals.build_schema_signals(
        retrieval_result=retrieval_result,
        filter_result=filter_result,
        extraction_result=extraction_result
    )

    # Step 9: Package everything into final output
    return {
        "schema_context": {
            "schema_text": schema_context.schema_text,
            "selected_schema": extraction_result.selected_schema,
            "tokens_estimate": schema_context.tokens_estimate
        },
        "schema_signals": schema_signals.values,
        "debug_info": {
            "metadata_source": "cache" if db_config.get("use_cache") else "live",
            "retrieval_result": retrieval_result,
            "filter_result": filter_result,
            "extraction_result": extraction_result,
            "rewrite_info": rewrite_info
        }
    }
