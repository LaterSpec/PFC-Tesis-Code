"""Step 2 of LinkAlign: filtering irrelevant schemas."""
from __future__ import annotations

from typing import Any, Dict, List

from .types import DatabaseMetadata, FilterResult, RetrievalResult


def filter_irrelevant_schemas(
    question: str,
    retrieval_result: RetrievalResult,
    databases: List[DatabaseMetadata],
    rewrite_info: Dict[str, Any],
    max_final_schemas: int = 3,
) -> FilterResult:
    """Filter schema candidates that are unlikely to answer the question."""
    initial_candidates = retrieval_result.candidates
    num_initial = len(initial_candidates)

    # For Process 1 with single DB, we keep everything
    # In multi-DB scenarios, this would apply score thresholds or LLM-based filtering
    filtered = initial_candidates[:max_final_schemas]
    num_kept = len(filtered)

    return FilterResult(
        filtered_candidates=filtered,
        stats={
            "num_initial": num_initial,
            "num_kept": num_kept,
            "num_removed": num_initial - num_kept,
            "fraction_removed": (num_initial - num_kept) / num_initial if num_initial > 0 else 0.0
        }
    )
