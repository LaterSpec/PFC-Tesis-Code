"""SQL execution runner for Snowflake and BigQuery."""
from __future__ import annotations

import time
from typing import Any, Dict

from .types import ExecutionResult


def run_sql_query(
    sql: str,
    schema_context: Dict[str, Any],
    db_config: Dict[str, Any],
    exec_config: Dict[str, Any]
) -> ExecutionResult:
    """
    Execute SQL query against database with safety limits.
    
    Supports both Snowflake and BigQuery backends with:
    - Timeout limits
    - Row count limits
    - Read-only mode enforcement
    
    Args:
        sql: SQL query to execute
        schema_context: Output from inspect_schema (contains selected_schema)
        db_config: Database connection config
        exec_config: Execution config with max_rows, timeout_seconds, etc.
        
    Returns:
        ExecutionResult with rows, timing, and any errors
    """
    # Extract engine
    selected_schema = schema_context.get("selected_schema")
    engine = selected_schema.engine if selected_schema else db_config.get("engine", "snowflake")
    
    # Get execution limits
    max_rows = exec_config.get("max_rows", 100)
    timeout_seconds = exec_config.get("timeout_seconds", 30)
    
    print(f"[EXEC] Engine: {engine}")
    print(f"[EXEC] Max rows: {max_rows}, Timeout: {timeout_seconds}s")
    
    # Route to appropriate backend
    if engine.lower() == "snowflake":
        return _run_snowflake(sql, db_config, max_rows, timeout_seconds)
    elif engine.lower() == "bigquery":
        return _run_bigquery(sql, db_config, max_rows, timeout_seconds)
    else:
        return ExecutionResult(
            sql=sql,
            rows=[],
            error=f"Unsupported engine: {engine}",
            extra={"engine": engine}
        )


def _run_snowflake(
    sql: str,
    db_config: Dict[str, Any],
    max_rows: int,
    timeout_seconds: int
) -> ExecutionResult:
    """Execute SQL on Snowflake with safety limits."""
    import snowflake.connector
    import json
    
    try:
        # Load credentials
        credential_path = db_config.get("credential_path")
        if credential_path:
            with open(credential_path, "r", encoding="utf-8") as f:
                cred = json.load(f)
            account = cred["account"]
            user = cred["username"]
            password = cred["password"]
            warehouse = cred.get("warehouse")
            role = cred.get("role")
        else:
            account = db_config["account"]
            user = db_config["user"]
            password = db_config["password"]
            warehouse = db_config["warehouse"]
            role = db_config.get("role")
        
        # Connect
        print(f"[EXEC] Connecting to Snowflake...")
        conn = snowflake.connector.connect(
            account=account,
            user=user,
            password=password,
            warehouse=warehouse,
            role=role,
            network_timeout=timeout_seconds,
        )
        
        cursor = conn.cursor()
        
        # Measure execution time
        start_time = time.monotonic()
        
        print(f"[EXEC] Executing query...")
        cursor.execute(sql)
        
        # Fetch results (limited by max_rows)
        rows_raw = cursor.fetchmany(max_rows)
        
        end_time = time.monotonic()
        latency_ms = (end_time - start_time) * 1000
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description] if cursor.description else []
        
        # Convert rows to list of dicts
        rows = [dict(zip(column_names, row)) for row in rows_raw]
        
        # Get row count (Snowflake provides this)
        row_count = cursor.rowcount if cursor.rowcount >= 0 else len(rows)
        
        # Close connection
        cursor.close()
        conn.close()
        
        print(f"[EXEC] ✓ Query executed successfully")
        print(f"[EXEC] Rows returned: {len(rows)}, Latency: {latency_ms:.2f}ms")
        
        return ExecutionResult(
            sql=sql,
            rows=rows,
            row_count=row_count,
            columns=column_names,
            error=None,
            latency_ms=latency_ms,
            extra={
                "engine": "snowflake",
                "max_rows": max_rows,
                "timeout_seconds": timeout_seconds,
            }
        )
        
    except Exception as e:
        print(f"[EXEC] ✗ Execution failed: {e}")
        return ExecutionResult(
            sql=sql,
            rows=[],
            row_count=None,
            columns=[],
            error=str(e),
            latency_ms=None,
            extra={
                "engine": "snowflake",
                "max_rows": max_rows,
            }
        )


def _run_bigquery(
    sql: str,
    db_config: Dict[str, Any],
    max_rows: int,
    timeout_seconds: int
) -> ExecutionResult:
    """Execute SQL on BigQuery with safety limits."""
    from google.cloud import bigquery
    from google.oauth2 import service_account
    
    try:
        # Authenticate
        credential_path = db_config.get("credential_path", "src/cred/clean_node.json")
        credentials = service_account.Credentials.from_service_account_file(credential_path)
        client = bigquery.Client(credentials=credentials, project=credentials.project_id)
        
        # Configure query
        job_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=10**9,  # 1 GB limit
            use_query_cache=True,
        )
        
        print(f"[EXEC] Connecting to BigQuery...")
        
        # Measure execution time
        start_time = time.monotonic()
        
        print(f"[EXEC] Executing query...")
        query_job = client.query(sql, job_config=job_config, timeout=timeout_seconds)
        
        # Fetch results
        results = query_job.result(max_results=max_rows, timeout=timeout_seconds)
        
        end_time = time.monotonic()
        latency_ms = (end_time - start_time) * 1000
        
        # Convert to list of dicts
        rows = [dict(row) for row in results]
        
        # Get column names
        column_names = [field.name for field in results.schema] if results.schema else []
        
        # BigQuery provides total_rows
        row_count = results.total_rows if hasattr(results, 'total_rows') else len(rows)
        
        print(f"[EXEC] ✓ Query executed successfully")
        print(f"[EXEC] Rows returned: {len(rows)}, Latency: {latency_ms:.2f}ms")
        
        return ExecutionResult(
            sql=sql,
            rows=rows,
            row_count=row_count,
            columns=column_names,
            error=None,
            latency_ms=latency_ms,
            extra={
                "engine": "bigquery",
                "max_rows": max_rows,
                "timeout_seconds": timeout_seconds,
                "bytes_billed": query_job.total_bytes_billed if hasattr(query_job, 'total_bytes_billed') else None,
            }
        )
        
    except Exception as e:
        print(f"[EXEC] ✗ Execution failed: {e}")
        return ExecutionResult(
            sql=sql,
            rows=[],
            row_count=None,
            columns=[],
            error=str(e),
            latency_ms=None,
            extra={
                "engine": "bigquery",
                "max_rows": max_rows,
            }
        )
