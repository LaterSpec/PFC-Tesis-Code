"""Prompt construction for SQL generation using LLM."""
from __future__ import annotations

from typing import Any, Dict, List


def build_generator_system_prompt(engine: str, gen_config: Dict[str, Any]) -> str:
    """
    Build the system prompt for SQL generation.
    
    Instructs the LLM on how to behave as a SQL generator, including:
    - Output format requirements (JSON)
    - Safety constraints (no DDL/DML)
    - Dialect specifics (BigQuery vs Snowflake)
    
    Args:
        engine: Database engine ("bigquery" or "snowflake")
        gen_config: Generation configuration dict
        
    Returns:
        System prompt string
    """
    output_format = gen_config.get("output_format", "json")
    
    dialect_notes = {
        "bigquery": "Use BigQuery SQL dialect. Tables must be quoted with backticks `project.dataset.table`.",
        "snowflake": (
            "Use Snowflake SQL dialect. Tables must be fully qualified as DATABASE.SCHEMA.TABLE, "
            "and every column or table identifier MUST be wrapped in double quotes exactly as shown "
            "in the schema context."
        ),
    }
    
    dialect_hint = dialect_notes.get(engine.lower(), "Use standard SQL.")
    
    system_prompt = f"""You are an expert SQL query generator for text-to-SQL systems.

Your task is to generate a single SQL query that answers the user's natural language question.

IMPORTANT RULES:
1. Use ONLY the tables and columns provided in the schema context
2. Do NOT use tables or columns that are not in the provided schema
3. {dialect_hint}
4. Output MUST be valid JSON in this exact format:
{{
  "sql": "your SQL query here",
  "expected_shape": {{
    "kind": "aggregation|list|scalar",
    "rows": "one|many|unknown"
  }},
  "rationale": "brief explanation of query logic"
}}

PROHIBITED:
- DDL statements (CREATE, DROP, ALTER, TRUNCATE)
- Unsafe DML (DELETE, UPDATE without WHERE/LIMIT)
- Multiple statements (only ONE SELECT query)
- Schema modifications
- System table access

QUERY BEST PRACTICES:
- Use explicit JOINs when combining tables
- Include WHERE clauses for filtering
- Use GROUP BY with aggregate functions (COUNT, SUM, AVG, etc.)
- Add ORDER BY for sorted results when relevant
- Use LIMIT when listing results to avoid huge outputs
- Prefer UPPER() or LOWER() for case-insensitive string matching

EXPECTED SHAPE:
- kind="aggregation": query uses COUNT, SUM, AVG, etc. Usually returns 1 row
- kind="list": query returns multiple rows (names, records, etc.)
- kind="scalar": query returns a single value without aggregation
- rows="one": expect exactly 1 row
- rows="many": expect multiple rows
- rows="unknown": unsure about row count

Return ONLY valid JSON. No markdown, no code fences, no extra text."""
    
    return system_prompt


def build_generator_user_prompt(question: str, schema_text: str) -> str:
    """
    Build the user prompt with question and schema context.
    
    Args:
        question: Natural language question from user
        schema_text: Serialized schema from inspect_schema module
        
    Returns:
        User prompt string
    """
    user_prompt = f"""Question: {question}

Schema context:
{schema_text}

Generate a SQL query that answers the question using ONLY the tables and columns shown above.
If the schema shows identifiers with double quotes, you MUST use the same quoted form in the SQL.
Return your response as valid JSON with the format specified in the system instructions."""
    
    return user_prompt


def build_generator_messages(
    question: str,
    schema_context: Dict[str, Any],
    gen_config: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Build chat-formatted messages for HuggingFace pipeline.
    
    Combines system and user prompts into the message format expected
    by chat-tuned models (Llama-3.2, Qwen, etc.).
    
    Args:
        question: Natural language question
        schema_context: Dict containing "schema_text" and optionally "selected_schema"
        gen_config: Generation configuration
        
    Returns:
        List of message dicts with "role" and "content" keys
    """
    # Extract schema text
    schema_text = schema_context.get("schema_text", "")
    
    # Extract engine from selected_schema if available
    selected_schema = schema_context.get("selected_schema")
    engine = selected_schema.engine if selected_schema else "snowflake"
    
    # Build prompts
    system_content = build_generator_system_prompt(engine, gen_config)
    user_content = build_generator_user_prompt(question, schema_text)
    
    # Format as chat messages
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    return messages
