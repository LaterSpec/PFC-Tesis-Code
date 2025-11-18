# exec_sql Module

Safe SQL execution with validation and signal generation for the Strategic Control Module (SCM).

## Overview

The `exec_sql` module is the third stage of the LinkAlign text-to-SQL pipeline:
1. **inspect_schema**: Load schema metadata and select relevant columns
2. **gen_sql**: Generate SQL from natural language question
3. **exec_sql**: Execute SQL safely and return results ← THIS MODULE

## Features

- **Pre-execution validation**: Safety checks to prevent dangerous queries
- **Row limits**: Configurable maximum rows to prevent excessive data transfer
- **Timeout protection**: Query timeout to prevent long-running queries
- **Performance measurement**: Accurate latency tracking with microsecond precision
- **Signal generation**: 20+ signals for SCM decision-making
- **Multi-engine support**: Snowflake and BigQuery backends

## Architecture

```
exec_sql/
├── __init__.py          # Main orchestrator (exec_sql function)
├── types.py             # ExecutionResult, ExecutionSignals dataclasses
├── runner.py            # SQL execution (_run_snowflake, _run_bigquery)
├── validation.py        # Pre-execution safety checks
├── signals.py           # Signal construction for SCM
└── prompts.py           # Optional LLM-based result explanation/auditing
```

## Usage

### Basic Usage

```python
from modules.exec_sql import exec_sql

result = exec_sql(
    sql="SELECT * FROM usa_names WHERE year=2000 LIMIT 10",
    schema_context=schema_ctx,  # From inspect_schema
    expected_shape={"kind": "list", "rows": "many"},  # From gen_sql
    db_config={"engine": "snowflake", "credential_file": "..."},
    exec_config={"max_rows": 100, "timeout": 30}
)

# Access results
print(result["execution"]["rows"])
print(result["execution"]["row_count"])
print(result["exec_signals"].values)
```

### Configuration

**db_config** (required):
```python
{
    "engine": "snowflake",  # or "bigquery"
    "credential_file": "path/to/snowflake_credential.json",
    # OR direct credentials:
    "username": "...",
    "password": "...",
    "account": "...",
    "warehouse": "...",
    "role": "..."
}
```

**exec_config** (optional):
```python
{
    "max_rows": 1000,           # Maximum rows to fetch (default: 1000)
    "timeout": 30,              # Query timeout in seconds (default: 30)
    "strict_validation": False  # Treat warnings as errors (default: False)
}
```

### Return Value

```python
{
    "sql": "SELECT ...",
    "validation": {
        "safe_to_execute": True,
        "errors": [],
        "warnings": ["Missing LIMIT clause"],
        "recommendations": ["Add LIMIT 1000"]
    },
    "execution": {
        "rows": [{...}, {...}],
        "row_count": 42,
        "columns": ["name", "count"],
        "error": None,
        "latency_ms": 123.45,
        "extra": {"engine": "snowflake", "max_rows": 1000}
    },
    "exec_signals": ExecutionSignals(values={
        "exec_success": 1.0,
        "exec_error": 0.0,
        "exec_latency_ms": 123.45,
        "row_count": 42.0,
        "truncated": 0.0,
        "shape_row_mismatch": 0.0,
        "result_quality_score": 0.9,
        ...  # 20+ more signals
    }),
    "debug_info": {...}
}
```

## Validation Checks

The module performs pre-execution validation to prevent:

- **DDL/DML operations**: No CREATE, DROP, ALTER, INSERT, UPDATE, DELETE
- **Dangerous keywords**: TRUNCATE, REPLACE, MERGE
- **Cartesian products**: Joins without ON/USING clauses
- **Missing LIMIT**: List queries without row limits

Warnings for:
- **SELECT \***: Recommend explicit column names
- **Missing LIMIT on list queries**: Could return too many rows

## Signal Generation

The module generates 20+ signals for SCM:

### Execution Status
- `exec_success`, `exec_error` (0/1)

### Performance
- `exec_latency_ms` (float)
- `latency_fast`, `latency_medium`, `latency_slow` (0/1)

### Row Counts
- `row_count` (float)
- `rows_empty`, `rows_single`, `rows_few`, `rows_many` (0/1)

### Truncation
- `truncated` (1 if max_rows hit)
- `truncation_warning` (1 if results may be incomplete)

### Shape Consistency
- `actual_shape_empty`, `actual_shape_scalar`, `actual_shape_list` (0/1)
- `expected_shape_aggregation`, `expected_shape_list`, `expected_shape_scalar` (0/1)
- `shape_row_mismatch` (1 if expected != actual)

### Quality
- `result_quality_score` (0.0-1.0 heuristic)
- `num_columns` (float)

### Engine
- `engine_snowflake`, `engine_bigquery` (0/1)

## Safety Features

1. **Read-only mode**: Only SELECT queries allowed
2. **Row limits**: Configurable maximum rows per query
3. **Timeout protection**: Queries terminated after timeout
4. **Validation before execution**: Catches dangerous patterns
5. **Error handling**: All exceptions caught and returned as ExecutionResult.error

## Examples

### Example 1: Simple Query

```python
result = exec_sql(
    sql="SELECT COUNT(*) as total FROM usa_names",
    schema_context=schema_ctx,
    expected_shape={"kind": "aggregation", "rows": "one"},
    db_config={"engine": "snowflake", "credential_file": "cred.json"}
)

# Check success
if result["execution"]["error"]:
    print(f"Error: {result['execution']['error']}")
else:
    print(f"Total: {result['execution']['rows'][0]['total']}")
```

### Example 2: List Query with Limit

```python
result = exec_sql(
    sql="SELECT name, SUM(number) as total FROM usa_names GROUP BY name ORDER BY total DESC LIMIT 10",
    schema_context=schema_ctx,
    expected_shape={"kind": "list", "rows": "many"},
    db_config={"engine": "snowflake", "credential_file": "cred.json"},
    exec_config={"max_rows": 20}  # Fetch up to 20 rows
)

for row in result["execution"]["rows"]:
    print(f"{row['name']}: {row['total']}")
```

### Example 3: Handling Validation Errors

```python
result = exec_sql(
    sql="DROP TABLE usa_names",  # Dangerous!
    schema_context=schema_ctx,
    expected_shape={"kind": "scalar", "rows": "one"},
    db_config={"engine": "snowflake", "credential_file": "cred.json"}
)

# Validation will catch this
assert not result["validation"]["safe_to_execute"]
assert result["execution"]["error"] is not None
print(result["validation"]["errors"])
# Output: ['Contains DDL keyword: DROP']
```

## Testing

Run the end-to-end demo:

```bash
# Default question
python link/demo_exec_sql_snowflake.py

# Custom question
python link/demo_exec_sql_snowflake.py --question "How many babies were named John in 1950?"

# Interactive mode
python link/demo_exec_sql_snowflake.py --interactive

# With configuration
python link/demo_exec_sql_snowflake.py --question "Top 10 names in 2000" --max-rows 10 --timeout 60
```

## Future Work

- **Result explanation**: LLM-based natural language summaries (prompts.py has templates)
- **Result auditing**: LLM validates if results answer question correctly
- **Query plan analysis**: Estimate cost and suggest optimizations
- **Caching**: Cache query results for repeated questions
- **Streaming**: Support large result sets with cursor-based iteration

## Integration with SCM

The execution signals are designed for SCM decision-making:

```python
# After exec_sql, SCM can decide:
if exec_signals.values["exec_error"] > 0:
    action = "repair"  # Fix SQL and retry
elif exec_signals.values["shape_row_mismatch"] > 0:
    action = "regenerate"  # Wrong shape, regenerate SQL
elif exec_signals.values["result_quality_score"] < 0.5:
    action = "review"  # Low quality, needs human review
else:
    action = "accept"  # Good result, proceed
```

## Dependencies

- `snowflake-connector-python>=4.0.0`
- `google-cloud-bigquery>=3.0.0` (optional, for BigQuery support)
- Python 3.8+

## License

See LICENSE file in repository root.
