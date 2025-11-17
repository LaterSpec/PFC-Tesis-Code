"""Future schema indexing helpers (embeddings, vector stores, etc.)."""
from __future__ import annotations

from typing import Any, Dict

from .types import DatabaseMetadata


def build_schema_index(metadata: Dict[str, Any], embed_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build a search index over tables/columns for retrieval."""

    raise NotImplementedError


def query_schema_index(index: Dict[str, Any], question: str, top_k: int = 5) -> Dict[str, Any]:
    """Query a schema index with a natural-language question."""

    raise NotImplementedError
