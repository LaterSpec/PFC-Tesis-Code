"""
Validation logic for SQL execution.
Checks for safety, engine-specific rules, and best practices.
"""
from __future__ import annotations

from typing import Any, Dict, List


def validate_for_execution(
    sql: str,
    engine: str,
    expected_shape: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Perform additional validation checks before executing SQL.
    
    Reuses validation from gen_sql.validation and adds execution-specific checks.
    
    Args:
        sql: SQL query to validate
        engine: Database engine ("snowflake" or "bigquery")
        expected_shape: Expected result shape from gen_sql
        
    Returns:
        Dictionary with:
            - safe_to_execute: bool
            - errors: List[str] (blocking issues)
            - warnings: List[str] (non-blocking concerns)
            - recommendations: List[str] (suggestions)
    """
    errors: List[str] = []
    warnings: List[str] = []
    recommendations: List[str] = []
    
    sql_clean = sql.strip()
    sql_upper = sql_clean.upper()
    
    # === BASIC SAFETY CHECKS ===
    
    if not sql_clean:
        errors.append("SQL is empty")
        return {
            "safe_to_execute": False,
            "errors": errors,
            "warnings": warnings,
            "recommendations": recommendations
        }
    
    # Must be SELECT only
    if not sql_upper.startswith("SELECT"):
        errors.append("Only SELECT queries are allowed for execution")
    
    # No DDL/DML
    dangerous_keywords = [
        "DROP", "TRUNCATE", "ALTER", "CREATE", "RENAME",
        "DELETE", "UPDATE", "INSERT", "MERGE"
    ]
    for keyword in dangerous_keywords:
        if f"\\b{keyword}\\b" in sql_upper:
            errors.append(f"Dangerous keyword detected: {keyword}")
    
    # === EXECUTION-SPECIFIC CHECKS ===
    
    # Check for LIMIT clause on list queries
    shape_kind = expected_shape.get("kind", "unknown")
    has_limit = "LIMIT" in sql_upper
    
    if shape_kind == "list" and not has_limit:
        warnings.append("List query without LIMIT - may return many rows")
        recommendations.append("Consider adding LIMIT clause to avoid large result sets")
    
    # Check for expensive operations
    has_cartesian_product = (
        " JOIN " in sql_upper and 
        " ON " not in sql_upper and 
        " USING " not in sql_upper
    )
    if has_cartesian_product:
        warnings.append("Possible cartesian product (JOIN without ON/USING)")
        errors.append("Cartesian products are too expensive - query rejected")
    
    # Check for SELECT * on potentially large tables
    if "SELECT *" in sql_upper and not has_limit:
        warnings.append("SELECT * without LIMIT may be slow")
        recommendations.append("Specify column names explicitly or add LIMIT")
    
    # === ENGINE-SPECIFIC CHECKS ===
    
    if engine.lower() == "snowflake":
        # Snowflake-specific validations
        # La BD y el schema se fijan en la conexión, no es obligatorio calificar en el SQL
        pass
    
    elif engine.lower() == "bigquery":
        # BigQuery-specific validations
        if "`" not in sql_clean:
            warnings.append("BigQuery tables should use backticks")
    
    # === FINAL DECISION ===
    
    safe_to_execute = len(errors) == 0
    
    return {
        "safe_to_execute": safe_to_execute,
        "errors": errors,
        "warnings": warnings,
        "recommendations": recommendations
    }