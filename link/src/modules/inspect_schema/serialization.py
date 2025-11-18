"""Serialize selected schema context to text for LLM prompts."""
from __future__ import annotations

from typing import Any, Dict

from .types import DatabaseMetadata, SchemaContext


def serialize_schema_context(
    selected_schema: DatabaseMetadata,
    serialization_config: Dict[str, Any],
) -> SchemaContext:
    """Convert selected schema into a compact prompt-friendly string."""
    include_types = serialization_config.get("include_types", True)
    include_descriptions = serialization_config.get("include_descriptions", True)

    lines = [
        f"You are working with a {selected_schema.engine} database: {selected_schema.db_id}",
        ""
    ]

    for table in selected_schema.tables:
        lines.append(f"Table: {table.full_name}")
        
        # Filter only needed columns
        needed_cols = [c for c in table.columns if c.extra.get("needed", False)]
        
        if needed_cols:
            lines.append("Relevant columns:")
            for col in needed_cols:
                if include_types:
                    col_line = f"    - \"{col.name}\" ({col.type})"
                else:
                    col_line = f"    - \"{col.name}\""

                if include_descriptions and col.description:
                    col_line += f": {col.description}"

                lines.append(col_line)
        else:
            # Fallback: if no columns marked as needed, show all
            lines.append("Columns:")
            for col in table.columns:
                if include_types:
                    col_line = f"    - \"{col.name}\" ({col.type})"
                else:
                    col_line = f"    - \"{col.name}\""
                lines.append(col_line)
        
        lines.append("")

    lines.append("Use only the specified tables and columns in your SQL query.")
    lines.append("Always reference Snowflake identifiers with double quotes, exactly as shown above.")
    lines.append("Fully qualify tables as DATABASE.SCHEMA.TABLE when writing SQL.")

    schema_text = "\n".join(lines)
    tokens_estimate = len(schema_text.split())  # Simple word count approximation

    return SchemaContext(
        schema_text=schema_text,
        selected_schema=selected_schema,
        tokens_estimate=tokens_estimate
    )