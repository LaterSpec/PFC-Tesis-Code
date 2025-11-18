"""Signal construction for Strategic Control Module (SCM)."""
from __future__ import annotations

from typing import Any, Dict

from .types import GenerationResult, GenerationSignals


def build_generation_signals(
    result: GenerationResult,
    aggregate_info: Dict[str, Any]
) -> GenerationSignals:
    """
    Build numerical signals for the Strategic Control Module.
    
    These signals help the SCM decide on actions:
    - Accept generated SQL
    - Request regeneration
    - Apply format restrictions
    - Escalate to repair module
    
    Args:
        result: GenerationResult from generator module
        aggregate_info: Validation statistics from validation module
        
    Returns:
        GenerationSignals with .values dict
    """
    values: Dict[str, float] = {}
    
    # === BASIC COUNTS ===
    values["num_candidates"] = float(aggregate_info.get("num_candidates", 0))
    values["num_format_ok"] = float(aggregate_info.get("num_format_ok", 0))
    values["primary_format_ok"] = 1.0 if aggregate_info.get("primary_format_ok", False) else 0.0
    
    # === PRIMARY CANDIDATE FEATURES ===
    primary = result.primary
    primary_validation = primary.extra.get("validation", {})
    primary_features = primary_validation.get("features", {})
    
    # SQL length
    values["primary_sql_length_tokens"] = primary_features.get("length_tokens", 0.0)
    
    # Query complexity indicators
    values["has_agg_func"] = primary_features.get("has_agg_func", 0.0)
    values["has_group_by"] = primary_features.get("has_group_by", 0.0)
    values["has_group_by_without_agg"] = primary_features.get("has_group_by_without_agg", 0.0)
    values["has_limit"] = primary_features.get("has_limit", 0.0)
    values["has_order_by"] = primary_features.get("has_order_by", 0.0)
    values["has_where"] = primary_features.get("has_where", 0.0)
    values["has_join"] = primary_features.get("has_join", 0.0)
    values["has_distinct"] = primary_features.get("has_distinct", 0.0)
    values["has_subquery"] = primary_features.get("has_subquery", 0.0)
    
    # === VALIDATION STATUS ===
    num_errors = len(primary_validation.get("errors", []))
    num_warnings = len(primary_validation.get("warnings", []))
    
    values["num_errors"] = float(num_errors)
    values["num_warnings"] = float(num_warnings)
    
    # === RISK SCORE (HEURISTIC) ===
    # Simple heuristic risk assessment based on validation and features
    # Higher score = more risky, should be reviewed or regenerated
    # Range: [0.0, 1.0]
    
    risk = 0.0
    
    # Base risk from format validation
    if not aggregate_info.get("primary_format_ok", False):
        risk += 0.5  # Major penalty for format failure
    
    # Add risk for errors/warnings
    risk += min(num_errors * 0.15, 0.3)  # Up to +0.3 for errors
    risk += min(num_warnings * 0.05, 0.15)  # Up to +0.15 for warnings
    
    # Structural issues
    if values["has_group_by_without_agg"] > 0:
        risk += 0.2  # GROUP BY without aggregation is suspicious
    
    # Length-based risk
    length_tokens = values["primary_sql_length_tokens"]
    if length_tokens > 300:
        risk += 0.15  # Very long queries are risky
    elif length_tokens < 10:
        risk += 0.1  # Very short queries might be incomplete
    
    # Complexity without filtering
    if values["has_join"] > 0 and values["has_where"] == 0:
        risk += 0.1  # JOIN without WHERE might be unfiltered cartesian product
    
    # Normalize to [0.0, 1.0]
    risk = min(max(risk, 0.0), 1.0)
    values["risk_score"] = risk
    
    # === CONFIDENCE SCORE (INVERSE OF RISK) ===
    # Higher confidence = lower risk, more likely to be correct
    values["confidence_score"] = 1.0 - risk
    
    # === EXPECTED SHAPE SIGNALS ===
    expected_shape = primary.expected_shape
    shape_kind = expected_shape.get("kind", "unknown")
    shape_rows = expected_shape.get("rows", "unknown")
    
    # One-hot encode shape kind
    values["shape_is_aggregation"] = 1.0 if shape_kind == "aggregation" else 0.0
    values["shape_is_list"] = 1.0 if shape_kind == "list" else 0.0
    values["shape_is_scalar"] = 1.0 if shape_kind == "scalar" else 0.0
    values["shape_is_unknown"] = 1.0 if shape_kind == "unknown" else 0.0
    
    # One-hot encode expected rows
    values["shape_rows_one"] = 1.0 if shape_rows == "one" else 0.0
    values["shape_rows_many"] = 1.0 if shape_rows == "many" else 0.0
    values["shape_rows_unknown"] = 1.0 if shape_rows == "unknown" else 0.0
    
    return GenerationSignals(values=values)
