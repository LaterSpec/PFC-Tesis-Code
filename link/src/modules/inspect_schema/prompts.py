"""Prompt templates for LLM-based schema extraction.

This module contains system and user prompts for the column auditor,
which decides what columns are needed to answer a natural language question.
"""
from __future__ import annotations

from typing import Dict, List

from .types import TableMetadata


def build_auditor_system_prompt() -> str:
    """Build the system prompt for the column auditor."""
    return (
        "You are a schema auditor for text-to-SQL systems.\n"
        "Given a natural language question and a database table schema, "
        "decide which columns are needed to answer the question.\n\n"
        "IMPORTANT RULES:\n"
        "1. Return ONLY a valid JSON object, no extra text before or after.\n"
        "2. Use only column names that appear in the provided schema.\n"
        "3. If a column is not needed, set needed to false.\n"
        "4. Include ALL columns from the schema in your response.\n\n"
        "Response format:\n"
        "{\n"
        '  "columns": [\n'
        '    {"name": "column_name", "needed": true, "reason": "short explanation"},\n'
        '    {"name": "another_column", "needed": false, "reason": "not mentioned"}\n'
        "  ]\n"
        "}"
    )


def build_auditor_user_prompt(question: str, table: TableMetadata) -> str:
    """Build the user prompt with the question and table schema."""
    # Build column list
    columns_text = []
    for col in table.columns:
        col_line = f"- {col.name} ({col.type})"
        if col.description:
            col_line += f": {col.description}"
        columns_text.append(col_line)
    
    return (
        f"Question: {question}\n\n"
        f"Table: {table.full_name}\n"
        "Columns:\n" +
        "\n".join(columns_text)
    )


def build_auditor_messages(
    question: str,
    table: TableMetadata,
    extraction_config: Dict[str, any]
) -> List[Dict[str, str]]:
    """Build the full message list for the LLM pipeline."""
    return [
        {
            "role": "system",
            "content": build_auditor_system_prompt()
        },
        {
            "role": "user",
            "content": build_auditor_user_prompt(question, table)
        }
    ]
