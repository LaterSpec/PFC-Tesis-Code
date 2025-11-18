"""Static SQL validation and guardrails (no execution)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .types import SqlCandidate


def validate_candidates(
    candidates: List[SqlCandidate],
    engine: str
) -> Tuple[List[SqlCandidate], Dict[str, Any]]:
    """
    Validate all SQL candidates with static checks.
    
    Performs lightweight validation without executing queries:
    - Format checks (starts with SELECT, has FROM, etc.)
    - Safety checks (no DDL/DML)
    - Feature extraction (GROUP BY, aggregations, LIMIT, etc.)
    
    Args:
        candidates: List of SqlCandidate objects
        engine: Database engine ("bigquery" or "snowflake")
        
    Returns:
        Tuple of:
            - List[SqlCandidate]: validated candidates (same list for now)
            - Dict[str, Any]: aggregate info for signals
    """
    per_candidate_info = []
    
    for i, candidate in enumerate(candidates):
        validation_result = validate_sql(candidate.sql, engine)
        
        # Store validation result in candidate extra
        candidate.extra["validation"] = validation_result
        
        per_candidate_info.append({
            "index": i,
            "format_ok": validation_result["format_ok"],
            "errors": validation_result["errors"],
            "warnings": validation_result["warnings"],
            "features": validation_result["features"],
        })
        
        if not validation_result["format_ok"]:
            print(f"[VALIDATION] Candidate {i} failed: {validation_result['errors']}")
    
    # Compute aggregate statistics
    num_candidates = len(candidates)
    num_format_ok = sum(1 for info in per_candidate_info if info["format_ok"])
    primary_format_ok = per_candidate_info[0]["format_ok"] if per_candidate_info else False
    
    aggregate_info = {
        "num_candidates": num_candidates,
        "num_format_ok": num_format_ok,
        "primary_format_ok": primary_format_ok,
        "per_candidate_features": per_candidate_info,
    }
    
    return candidates, aggregate_info


def validate_sql(sql: str, engine: str) -> Dict[str, Any]:
    """
    Validate a single SQL query with static checks.
    
    Args:
        sql: SQL query string
        engine: Database engine ("bigquery" or "snowflake")
        
    Returns:
        Dictionary with:
            - format_ok: bool (overall pass/fail)
            - errors: List[str] (blocking issues)
            - warnings: List[str] (non-blocking issues)
            - features: Dict[str, float] (extracted features for signals)
    """
    sql_clean = sql.strip()
    sql_upper = sql_clean.upper()
    
    errors: List[str] = []
    warnings: List[str] = []
    features: Dict[str, float] = {}
    
    # === BASIC FORMAT CHECKS ===
    
    if not sql_clean:
        errors.append("SQL is empty")
        return {
            "format_ok": False,
            "errors": errors,
            "warnings": warnings,
            "features": features
        }
    
    # Must start with SELECT
    if not sql_upper.startswith("SELECT"):
        errors.append("SQL must start with SELECT")
    
    # Must contain FROM
    if "FROM" not in sql_upper:
        errors.append("SQL must contain FROM clause")
    
    # === SAFETY CHECKS (DDL/DML) ===
    
    dangerous_keywords = [
        "DROP", "TRUNCATE", "ALTER", "CREATE", "RENAME",
        "DELETE", "UPDATE", "INSERT", "MERGE"
    ]
    
    for keyword in dangerous_keywords:
        # Use word boundary to avoid false positives (e.g., "DROPPED" in column name)
        pattern = r'\b' + keyword + r'\b'
        if re.search(pattern, sql_upper):
            errors.append(f"Dangerous keyword detected: {keyword}")
    
    # === FEATURE EXTRACTION ===
    
    # Aggregation functions
    agg_functions = ["COUNT", "SUM", "AVG", "MAX", "MIN", "STDDEV", "VARIANCE"]
    has_agg_func = any(f"\\b{fn}\\b" in sql_upper and re.search(rf'\b{fn}\s*\(', sql_upper) 
                       for fn in agg_functions)
    features["has_agg_func"] = 1.0 if has_agg_func else 0.0
    
    # GROUP BY clause
    has_group_by = bool(re.search(r'\bGROUP\s+BY\b', sql_upper))
    features["has_group_by"] = 1.0 if has_group_by else 0.0
    
    # GROUP BY without aggregation (potential issue)
    has_group_by_without_agg = has_group_by and not has_agg_func
    features["has_group_by_without_agg"] = 1.0 if has_group_by_without_agg else 0.0
    
    if has_group_by_without_agg:
        warnings.append("GROUP BY without aggregation function")
    
    # LIMIT clause
    has_limit = bool(re.search(r'\bLIMIT\b', sql_upper))
    features["has_limit"] = 1.0 if has_limit else 0.0
    
    # ORDER BY clause
    has_order_by = bool(re.search(r'\bORDER\s+BY\b', sql_upper))
    features["has_order_by"] = 1.0 if has_order_by else 0.0
    
    # WHERE clause
    has_where = bool(re.search(r'\bWHERE\b', sql_upper))
    features["has_where"] = 1.0 if has_where else 0.0
    
    # JOIN detection
    has_join = bool(re.search(r'\b(INNER|LEFT|RIGHT|FULL|CROSS)?\s*JOIN\b', sql_upper))
    features["has_join"] = 1.0 if has_join else 0.0
    
    # DISTINCT
    has_distinct = bool(re.search(r'\bDISTINCT\b', sql_upper))
    features["has_distinct"] = 1.0 if has_distinct else 0.0
    
    # Subquery detection
    has_subquery = sql_clean.count("(") > sql_clean.count("COUNT(") + sql_clean.count("SUM(")
    features["has_subquery"] = 1.0 if has_subquery else 0.0
    
    # Token count estimation (rough)
    tokens = len(sql_clean.split())
    features["length_tokens"] = float(tokens)
    
    # Length check
    if tokens > 500:
        warnings.append(f"SQL is unusually long ({tokens} tokens)")
    
    # Multiple statements check (rough)
    semicolons = sql_clean.count(";")
    if semicolons > 1 or (semicolons == 1 and not sql_clean.endswith(";")):
        errors.append("Multiple SQL statements detected (only one SELECT allowed)")
    
    # === ENGINE-SPECIFIC CHECKS ===
    
    if engine.lower() == "bigquery":
        # BigQuery uses backticks for table names
        if "`" not in sql_clean and "." in sql_clean:
            warnings.append("BigQuery table names should use backticks")
    
    elif engine.lower() == "snowflake":
        # Snowflake uses DATABASE.SCHEMA.TABLE format
        # No special warnings for now
        pass
    
    # === OVERALL VALIDATION ===
    
    format_ok = len(errors) == 0
    
    return {
        "format_ok": format_ok,
        "errors": errors,
        "warnings": warnings,
        "features": features
    }
