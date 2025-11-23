"""Signal construction for Strategic Control Module (SCM)."""
from __future__ import annotations

from typing import Any, Dict

from .types import ExecutionResult, ExecutionSignals


def _is_semantically_empty(exec_result: ExecutionResult, expected_shape: Dict[str, Any]) -> bool:
    """
    Check if result is semantically empty even if physical rows exist.
    
    Examples:
    - COUNT(*) = 0
    - SUM(column) = 0 or NULL
    - Single row with all NULL values
    
    Args:
        exec_result: Execution result to check
        expected_shape: Expected shape from gen_sql
        
    Returns:
        True if result is semantically empty
    """
    # If no rows, it's physically empty
    if exec_result.row_count == 0 or not exec_result.rows:
        return True
    
    # For scalar/aggregation queries with single row
    shape_kind = expected_shape.get("kind", "unknown")
    if shape_kind in ["scalar", "aggregation"] and exec_result.row_count == 1:
        row = exec_result.rows[0]
        
        # Check if all values are 0, NULL, or empty
        if isinstance(row, dict):
            values = list(row.values())
        else:
            values = [row] if not isinstance(row, (list, tuple)) else row
        
        # Consider empty if:
        # - All values are None
        # - All values are 0 (for COUNT queries)
        # - All values are empty strings
        all_null = all(v is None for v in values)
        all_zero = all(v == 0 for v in values)
        all_empty_str = all(v == "" for v in values)
        
        return all_null or all_zero or all_empty_str
    
    return False


def build_execution_signals(
    exec_result: ExecutionResult,
    expected_shape: Dict[str, Any]
) -> ExecutionSignals:
    """
    Build numerical signals from execution result for SCM.
    
    These signals help the SCM decide on actions:
    - Accept the result
    - Request re-generation with different constraints
    - Flag for human review
    
    Args:
        exec_result: ExecutionResult from runner
        expected_shape: Expected result shape from gen_sql
        
    Returns:
        ExecutionSignals with .values dict
    """
    values: Dict[str, float] = {}
    
    # === EXECUTION STATUS ===
    values["exec_error"] = 1.0 if exec_result.error else 0.0
    values["exec_success"] = 0.0 if exec_result.error else 1.0
    
    # === PERFORMANCE METRICS ===
    values["exec_latency_ms"] = float(exec_result.latency_ms or 0.0)
    
    # Latency categories (for easier decision-making)
    latency = exec_result.latency_ms or 0.0
    values["latency_fast"] = 1.0 if latency < 1000 else 0.0  # < 1s
    values["latency_medium"] = 1.0 if 1000 <= latency < 5000 else 0.0  # 1-5s
    values["latency_slow"] = 1.0 if latency >= 5000 else 0.0  # > 5s
    
    # === ROW COUNT ANALYSIS ===
    row_count = exec_result.row_count if exec_result.row_count is not None else len(exec_result.rows)
    values["row_count"] = float(row_count)
    
    # Check for semantic emptiness (e.g., COUNT(*) = 0, SUM = 0 or NULL)
    is_semantically_empty = _is_semantically_empty(exec_result, expected_shape)
    
    # Row count categories
    values["rows_empty"] = 1.0 if (row_count == 0 or is_semantically_empty) else 0.0
    values["rows_semantically_empty"] = 1.0 if is_semantically_empty else 0.0  # For debugging
    values["rows_single"] = 1.0 if row_count == 1 else 0.0
    values["rows_few"] = 1.0 if 2 <= row_count <= 10 else 0.0
    values["rows_many"] = 1.0 if row_count > 10 else 0.0
    
    # === TRUNCATION ===
    max_rows = exec_result.extra.get("max_rows", 0)
    if max_rows > 0 and row_count >= max_rows:
        values["truncated"] = 1.0
        values["truncation_warning"] = 1.0
    else:
        values["truncated"] = 0.0
        values["truncation_warning"] = 0.0
    
    # === SHAPE CONSISTENCY ===
    shape_kind = expected_shape.get("kind", "unknown")
    shape_rows = expected_shape.get("rows", "unknown")
    
    # One-hot encode actual shape kind (based on row count)
    if row_count == 0:
        values["actual_shape_empty"] = 1.0
    elif row_count == 1:
        values["actual_shape_scalar"] = 1.0
    else:
        values["actual_shape_list"] = 1.0
    
    # Check consistency between expected and actual
    shape_mismatch = 0.0
    
    if shape_rows == "one" and row_count != 1:
        shape_mismatch = 1.0
    elif shape_rows == "many" and row_count <= 1:
        shape_mismatch = 1.0
    
    values["shape_row_mismatch"] = shape_mismatch
    
    # Expected shape kind (from gen_sql)
    values["expected_shape_aggregation"] = 1.0 if shape_kind == "aggregation" else 0.0
    values["expected_shape_list"] = 1.0 if shape_kind == "list" else 0.0
    values["expected_shape_scalar"] = 1.0 if shape_kind == "scalar" else 0.0
    
    # === RESULT QUALITY SCORE ===
    # Heuristic score combining multiple factors
    quality = 1.0
    
    # Penalty for errors
    if exec_result.error:
        quality = 0.0
    else:
        # Penalty for empty results (physical or semantic)
        if row_count == 0 or is_semantically_empty:
            quality -= 0.3
        
        # Penalty for shape mismatch
        if shape_mismatch > 0:
            quality -= 0.2
        
        # Penalty for truncation
        if values["truncated"] > 0:
            quality -= 0.1
        
        # Penalty for very slow queries
        if latency >= 10000:  # > 10s
            quality -= 0.2
        
        # Bonus for fast queries
        if latency < 500:  # < 0.5s
            quality += 0.1
    
    # Normalize to [0.0, 1.0]
    values["result_quality_score"] = max(min(quality, 1.0), 0.0)
    
    # === COLUMN COUNT ===
    values["num_columns"] = float(len(exec_result.columns))
    
    # === ENGINE INFO ===
    engine = exec_result.extra.get("engine", "unknown")
    values["engine_snowflake"] = 1.0 if engine == "snowflake" else 0.0
    values["engine_bigquery"] = 1.0 if engine == "bigquery" else 0.0
    
    return ExecutionSignals(values=values)
