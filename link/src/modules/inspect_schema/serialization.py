"""Serialize selected schema context to text for LLM prompts."""
from __future__ import annotations

from typing import Any, Dict

from .types import DatabaseMetadata, SchemaContext


def serialize_schema_context(
    selected_schema: DatabaseMetadata,
    serialization_config: Dict[str, Any],
) -> SchemaContext:
    """Convert selected schema into a compact prompt-friendly string."""
    style = serialization_config.get("style", "compact")
    include_types = serialization_config.get("include_types", True)
    include_descriptions = serialization_config.get("include_descriptions", True)

    lines = []
    lines.append(f"You are working with a {selected_schema.engine} database: {selected_schema.db_id}")
    lines.append("")

    for table in selected_schema.tables:
        lines.append(f"Table: {table.full_name}")
        if include_descriptions and table.description:
            lines.append(f"  Description: {table.description}")

        # Filter to only needed columns if available
        needed_cols = [col for col in table.columns if col.extra.get("needed", True)]
        if not needed_cols:
            needed_cols = table.columns  # Fallback to all if none marked

        lines.append("  Relevant columns:")
        for col in needed_cols:
            if include_types:
                col_line = f"    - {col.name} ({col.type})"
            else:
                col_line = f"    - {col.name}"

            if include_descriptions and col.description:
                col_line += f": {col.description}"

            lines.append(col_line)
        lines.append("")

    lines.append("Use only the specified tables and columns in your SQL query.")

    schema_text = "\n".join(lines)
    tokens_estimate = len(schema_text.split())  # Simple word count approximation

    return SchemaContext(
        schema_text=schema_text,
        selected_schema=selected_schema,
        tokens_estimate=tokens_estimate
    )
