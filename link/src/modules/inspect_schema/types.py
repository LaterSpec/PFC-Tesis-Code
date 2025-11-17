"""Shared dataclasses and type aliases for the inspect_schema pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ColumnMetadata:
    """Represents physical column metadata fetched from the backing warehouse."""

    name: str
    type: str
    description: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TableMetadata:
    """Represents a logical table and its fully qualified name."""

    table_name: str
    full_name: str
    columns: List[ColumnMetadata]
    description: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatabaseMetadata:
    """Represents a logical database (project.dataset for BigQuery)."""

    db_id: str
    engine: str
    tables: List[TableMetadata]
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Captures the output of the Step 1 retrieval phase."""

    candidates: List[Dict[str, Any]]
    round_id: int = 0
    debug_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterResult:
    """Carries surviving database candidates after filtering."""

    filtered_candidates: List[Dict[str, Any]]
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Holds the schema subset considered relevant for the question."""

    selected_schema: DatabaseMetadata
    coverage_stats: Dict[str, Any] = field(default_factory=dict)
    debug_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemaContext:
    """Textual schema block and metadata destined for the SQL generation prompt."""

    schema_text: str
    selected_schema: DatabaseMetadata
    tokens_estimate: Optional[int] = None


@dataclass
class SchemaSignals:
    """Numeric telemetry derived from the inspect_schema pipeline."""

    values: Dict[str, float] = field(default_factory=dict)
