"""Step 1 of LinkAlign: schema retrieval and query rewrites."""
from __future__ import annotations

from typing import Any, Dict, List

from .types import DatabaseMetadata, RetrievalResult


def initial_retrieval(
    question: str,
    databases: List[DatabaseMetadata],
    index: Dict[str, Any],
    max_candidates: int = 5,
) -> RetrievalResult:
    """Return initial database candidates for the supplied question."""
    # In Process 1, we have only one database, so retrieval is trivial
    # In future multi-DB scenarios, this would use semantic search over the index
    candidates = []
    for db in databases[:max_candidates]:
        candidates.append({
            "db_id": db.db_id,
            "score": 1.0  # Trivial score for single-DB case
        })

    return RetrievalResult(
        candidates=candidates,
        round_id=0,
        debug_info={"question": question, "num_databases": len(databases)}
    )


def rewrite_and_score_queries(
    question: str,
    retrieval_result: RetrievalResult,
    databases: List[DatabaseMetadata],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate query rewrites and scores per database candidate."""
    # For Process 1, we don't do query rewriting yet
    # Return the original question as the only rewrite
    return {
        "rewrites": [
            {
                "q_id": "Q0",
                "text": question,
                "score": 1.0
            }
        ]
    }
