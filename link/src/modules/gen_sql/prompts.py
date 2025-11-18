"""Prompt construction for SQL generation using LLM."""
from __future__ import annotations
from typing import Any, Dict, List

def build_generator_system_prompt(engine: str, gen_config: Dict[str, Any]) -> str:
    """
    Build the system prompt for SQL generation.
    """
    dialect_notes = {
        "snowflake": (
            "Use Snowflake SQL dialect. "
            "CRITICAL RULES FOR SNOWFLAKE:\n"
            "1. ALWAYS wrap every column name and table alias in double quotes (e.g., \"name\", \"state\").\n"
            "2. Use the FULLY QUALIFIED table name exactly as shown in the schema (e.g., USA_NAMES.USA_NAMES.TABLE_NAME).\n"
            "3. Do NOT wrap the SQL query string in markdown or code blocks inside the JSON value."
        ),
        "bigquery": "Use BigQuery SQL dialect. Use backticks for identifiers.",
    }
    
    dialect_hint = dialect_notes.get(engine.lower(), "Use standard SQL.")
    
    system_prompt = f"""You are an expert SQL query generator.

Your task is to generate a single SQL query that answers the user's natural language question using ONLY the provided schema.

{dialect_hint}

OUTPUT FORMAT REQUIREMENTS:
You MUST return a valid JSON object. Do NOT add any text outside the JSON.
{{
  "sql": "SELECT \\"column\\" FROM DB.SCHEMA.TABLE ...",
  "expected_shape": {{
    "kind": "aggregation|list|scalar",
    "rows": "one|many|unknown"
  }},
  "rationale": "Brief explanation"
}}

IMPORTANT FORMATTING RULES:
- The "sql" field must be a raw string containing the query.
- Do NOT start the SQL string with 'sql' or 'code'.
- Do NOT use markdown (```) inside the "sql" value.
- Ensure all quotes inside the SQL string are properly escaped if needed for JSON validity.

PROHIBITED:
- DDL (CREATE, DROP) or DML (INSERT, UPDATE).
- Selecting columns not present in the schema.
"""
    return system_prompt


def build_generator_user_prompt(question: str, schema_text: str) -> str:
    """
    Build the user prompt with question and schema context.
    """
    user_prompt = f"""Question: {question}

Schema Context (Available Tables and Columns):
{schema_text}

Instructions:
Generate the SQL query acting as a Snowflake expert.
1. Look at the schema above. If the table is 'USA_NAMES.USA_NAMES.USA_1910_CURRENT', use that EXACT full name.
2. Wrap ALL column names in double quotes (e.g., "year", "gender").
3. Return ONLY the JSON object.
"""
    return user_prompt


def build_generator_messages(
    question: str,
    schema_context: Dict[str, Any],
    gen_config: Dict[str, Any]
) -> List[Dict[str, str]]:
    """
    Build chat-formatted messages for HuggingFace pipeline.
    """
    schema_text = schema_context.get("schema_text", "")
    selected_schema = schema_context.get("selected_schema")
    engine = selected_schema.engine if selected_schema else "snowflake"
    
    system_content = build_generator_system_prompt(engine, gen_config)
    user_content = build_generator_user_prompt(question, schema_text)
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    return messages