"""Signal construction for Strategic Control Module (SCM)."""
from __future__ import annotations

from typing import Any, Dict

from .types import ExecutionResult, ExecutionSignals


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
    
    # Row count categories
    values["rows_empty"] = 1.0 if row_count == 0 else 0.0
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
        # Penalty for empty results (might be unexpected)
        if row_count == 0:
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
