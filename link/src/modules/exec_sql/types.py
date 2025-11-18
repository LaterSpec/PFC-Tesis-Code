"""Type definitions for exec_sql module."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionResult:
    """
    Result of executing SQL against the database.
    
    Attributes:
        sql: The SQL query that was executed
        rows: Sample of result rows (first N rows as dicts)
        row_count: Total number of rows returned (if known)
        columns: List of column names in result
        error: Error message if execution failed
        latency_ms: Execution time in milliseconds
        extra: Additional metadata (engine, max_rows, etc.)
    """
    sql: str
    rows: List[Dict[str, Any]] = field(default_factory=list)
    row_count: Optional[int] = None
    columns: List[str] = field(default_factory=list)
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionSignals:
    """
    Numerical signals from SQL execution for Strategic Control Module (SCM).
    
    These signals help the SCM decide whether to:
    - Accept the result
    - Request re-generation with different parameters
    - Flag for human review
    
    Attributes:
        values: Dictionary of signal_name -> float_value
            Standard signals include:
            - exec_error: 0.0 (success) or 1.0 (failed)
            - exec_latency_ms: execution time
            - row_count: number of rows returned
            - truncated: 1.0 if result was truncated by max_rows
            - shape_kind_*: one-hot encoding of result type
            - shape_row_mismatch: 1.0 if actual rows != expected
            - result_quality_score: heuristic quality [0.0, 1.0]
    """
    values: Dict[str, float] = field(default_factory=dict)
