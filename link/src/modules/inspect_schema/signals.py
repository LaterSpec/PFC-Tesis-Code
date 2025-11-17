"""Generate schema-linking signals for the SCM."""
from __future__ import annotations

from typing import Dict

from .types import ExtractionResult, FilterResult, RetrievalResult, SchemaSignals


def build_schema_signals(
    retrieval_result: RetrievalResult,
    filter_result: FilterResult,
    extraction_result: ExtractionResult,
) -> SchemaSignals:
    """Produce SCM-friendly numeric signals summarizing inspect_schema."""
    filter_stats = filter_result.stats
    coverage_stats = extraction_result.coverage_stats

    num_db_initial = filter_stats.get("num_initial", 0)
    num_db_kept = filter_stats.get("num_kept", 0)
    num_db_removed = filter_stats.get("num_removed", 0)

    num_columns_total = coverage_stats.get("num_columns_total", 0)
    num_columns_needed = coverage_stats.get("num_columns_needed", 0)

    # Calculate average score from filtered candidates
    avg_score = 0.0
    if filter_result.filtered_candidates:
        scores = [cand.get("score", 0.0) for cand in filter_result.filtered_candidates]
        avg_score = sum(scores) / len(scores)

    return SchemaSignals(
        values={
            "num_retrieval_rounds": retrieval_result.round_id + 1,
            "num_db_initial": float(num_db_initial),
            "num_db_kept": float(num_db_kept),
            "fraction_db_removed": num_db_removed / num_db_initial if num_db_initial > 0 else 0.0,
            "num_columns_total": float(num_columns_total),
            "num_columns_selected": float(num_columns_needed),
            "fraction_columns_selected": num_columns_needed / num_columns_total if num_columns_total > 0 else 0.0,
            "avg_schema_score": round(avg_score, 3)
        }
    )
