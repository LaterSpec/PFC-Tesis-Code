"""
exec_sql module: Safe SQL execution with validation and signal generation.

This module executes SQL queries against Snowflake or BigQuery with:
- Pre-execution safety validation
- Row count and timeout limits
- Performance measurement
- Signal generation for SCM

Usage:
    from modules.exec_sql import exec_sql
    
    result = exec_sql(
        sql="SELECT * FROM usa_names WHERE year=2000 LIMIT 10",
        schema_context=schema_ctx,
        expected_shape={"kind": "list", "rows": "many"},
        db_config={"engine": "snowflake", "credential_file": "..."},
        exec_config={"max_rows": 100, "timeout": 30}
    )
    
    print(result["execution"]["rows"])
    print(result["exec_signals"].values)
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .runner import run_sql_query
from .signals import build_execution_signals
from .types import ExecutionResult, ExecutionSignals
from .validation import validate_for_execution


def _adapt_sql_for_snowflake(sql: str, schema_context: Dict[str, Any]) -> str:
    """
    Adapta el SQL para Snowflake:
    - Pone comillas dobles a TODOS los nombres de columnas conocidos.
    - Pone comillas dobles a los nombres de tabla conocidos.
    No toca literales de texto entre comillas simples.
    """
    selected_schema = schema_context.get("selected_schema") if isinstance(schema_context, dict) else None
    if selected_schema is None:
        return sql

    # Recoger nombres de columnas y tablas desde el schema
    column_names = set()
    table_names = set()

    for table in getattr(selected_schema, "tables", []):
        table_names.add(table.table_name)
        for col in getattr(table, "columns", []):
            column_names.add(col.name)

    if not column_names and not table_names:
        return sql

    # Partir por literales de texto para no tocarlos
    segments = re.split(r"('(?:''|[^'])*')", sql)

    # Ordenar columnas por longitud para evitar que "year" rompa "year_total", etc.
    sorted_columns = sorted(column_names, key=len, reverse=True)
    sorted_tables = sorted(table_names, key=len, reverse=True)

    for idx in range(0, len(segments), 2):  # sólo partes fuera de comillas simples
        segment = segments[idx]

        # 1) Comillas para columnas
        for col in sorted_columns:
            # si NO está ya entre comillas dobles
            pattern = re.compile(rf'(?<!")\b{re.escape(col)}\b(?!")', re.IGNORECASE)
            segment = pattern.sub(f'"{col}"', segment)

        # 2) Comillas para tablas
        for tname in sorted_tables:
            pattern_table = re.compile(rf'(?<!")\b{re.escape(tname)}\b(?!")', re.IGNORECASE)
            segment = pattern_table.sub(f'"{tname}"', segment)

        segments[idx] = segment

    return "".join(segments)


def exec_sql(
    sql: str,
    schema_context: Dict[str, Any],
    expected_shape: Dict[str, Any],
    db_config: Dict[str, Any],
    exec_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute SQL query with validation, limits, and signal generation.
    
    Pipeline:
    1. Pre-execution validation (safety checks)
    2. Execute query with limits
    3. Generate execution signals for SCM
    
    Args:
        sql: SQL query to execute
        schema_context: Schema context from inspect_schema (contains db_id, etc.)
        expected_shape: Expected result shape from gen_sql
            {"kind": "aggregation"|"list"|"scalar", "rows": "one"|"many"}
        db_config: Database configuration
            - engine: "snowflake" or "bigquery"
            - For Snowflake: credential_file or username/password/account/warehouse/role
            - For BigQuery: project_id, credentials_file
        exec_config: Execution configuration (optional)
            - max_rows: Maximum rows to fetch (default: 1000)
            - timeout: Query timeout in seconds (default: 30)
            - strict_validation: Fail on validation warnings (default: False)
    
    Returns:
        Dict with:
        - sql: Original SQL query
        - validation: Dict with safe_to_execute, errors, warnings, recommendations
        - execution: Dict with rows, row_count, columns, error, latency_ms, extra
        - exec_signals: ExecutionSignals object with .values dict
        - debug_info: Optional debugging information
    """
    if exec_config is None:
        exec_config = {}
    
    engine = db_config.get("engine", "snowflake").lower()

    if engine == "snowflake":
        sql = _adapt_sql_for_snowflake(sql, schema_context)

    # Extract config
    max_rows = exec_config.get("max_rows", 1000)
    timeout = exec_config.get("timeout", 30)
    strict_validation = exec_config.get("strict_validation", False)
    
    # === STEP 1: PRE-EXECUTION VALIDATION ===
    print("[1/3] Validating SQL for execution...")
    
    validation = validate_for_execution(
        sql=sql,
        expected_shape=expected_shape,
        engine=db_config.get("engine", "snowflake")
    )
    
    # Check if safe to execute
    if not validation["safe_to_execute"]:
        print(f"[ERROR] SQL failed safety validation:")
        for error in validation["errors"]:
            print(f"  - {error}")
        
        # Return early with error
        return {
            "sql": sql,
            "validation": validation,
            "execution": {
                "rows": [],
                "row_count": 0,
                "columns": [],
                "error": "Safety validation failed",
                "latency_ms": 0,
                "extra": {}
            },
            "exec_signals": ExecutionSignals(values={"exec_error": 1.0}),
            "debug_info": {}
        }
        
    if strict_validation and validation["warnings"]:
        print(f"[ERROR] SQL failed strict validation (warnings present):")
        for warning in validation["warnings"]:
            print(f"  - {warning}")
            
        return {
            "sql": sql,
            "validation": validation,
            "execution": {
                "rows": [],
                "row_count": 0,
                "columns": [],
                "error": "Strict validation failed",
                "latency_ms": 0,
                "extra": {}
            },
            "exec_signals": ExecutionSignals(values={"exec_error": 1.0}),
            "debug_info": {}
        }

    # === STEP 2: EXECUTE SQL ===
    print(f"[2/3] Executing SQL (max_rows={max_rows}, timeout={timeout}s)...")
    
    exec_result = run_sql_query(
        sql=sql,
        schema_context=schema_context,
        db_config=db_config,
        exec_config=exec_config  # ✅ Ahora es un dict completo
    )
    
    if exec_result.error:
        print(f"[ERROR] Execution failed: {exec_result.error}")
    else:
        print(f"[✓] Execution successful: {exec_result.row_count} rows, {exec_result.latency_ms:.2f}ms")
    
    # === STEP 3: GENERATE SIGNALS ===
    print("[3/3] Generating execution signals for SCM...")
    
    exec_signals = build_execution_signals(exec_result, expected_shape)
    print(f"[✓] Generated {len(exec_signals.values)} signals")
    
    # === RETURN RESULTS ===
    return {
        "sql": sql,
        "validation": validation,
        "execution": {
            "rows": exec_result.rows,
            "row_count": exec_result.row_count,
            "columns": exec_result.columns,
            "error": exec_result.error,
            "latency_ms": exec_result.latency_ms,
            "extra": exec_result.extra
        },
        "exec_signals": exec_signals,
        "debug_info": {
            "max_rows": max_rows,
            "timeout": timeout,
            "strict_validation": strict_validation,
            "truncated": exec_signals.values.get("truncated", 0.0) > 0
        }
    }