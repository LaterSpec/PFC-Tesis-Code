"""
Detection logic for identifying issues in SQL queries.

This module analyzes the combination of:
- Natural language question
- Generated SQL
- Schema metadata
- Execution results

To detect potential semantic issues that could be repaired.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

from ..inspect_schema.types import ColumnMetadata, SchemaContext, TableMetadata
from .types import RepairInput, RepairIssue

# Regex patterns for detection
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
WHERE_SPLIT_PATTERN = re.compile(r"\bWHERE\b", re.IGNORECASE)
CLAUSE_BOUNDARY_PATTERN = re.compile(r"\b(GROUP\s+BY|ORDER\s+BY|LIMIT)\b", re.IGNORECASE)
FILTER_PATTERN = re.compile(
    r'(?P<column>"?[A-Za-z_][\w$]*"?)\s*(?P<operator>=)\s*(?P<value>"[^"]*"|\'[^\']*\'|\d+)',
    re.IGNORECASE,
)


def detect_issues(repair_input: RepairInput) -> List[RepairIssue]:
    """
    Main entry point: detect all issues in the repair input.
    
    Args:
        repair_input: Complete repair context
        
    Returns:
        List of detected issues
    """
    issues: List[RepairIssue] = []
    
    # Detect missing year filters
    issues.extend(_detect_missing_year_filter(repair_input))
    
    # Detect enum value mismatches
    issues.extend(_detect_enum_value_mismatches(repair_input))
    
    return issues


def _detect_missing_year_filter(repair_input: RepairInput) -> List[RepairIssue]:
    """
    Detect if question mentions a year but SQL doesn't filter by it.
    
    Args:
        repair_input: Repair context
        
    Returns:
        List with one RepairIssue if year filter is missing, empty otherwise
    """
    # Extract year from question
    year = _extract_question_year(repair_input.question)
    if year is None:
        return []
    
    # Check if SQL already has a year filter
    if re.search(r'\byear\b', repair_input.original_sql, re.IGNORECASE):
        return []
    
    # Look for a year column in schema
    column_info = _find_column(repair_input.schema_context, "year")
    if column_info is None:
        return []
    
    table_name, column = column_info
    
    return [
        RepairIssue(
            issue_type="missing_year_filter",
            column=getattr(column, "name", getattr(column, "column_name", "year")),
            table=table_name,
            question_value=year,
            details={"year": year},
        )
    ]


def _detect_enum_value_mismatches(repair_input: RepairInput) -> List[RepairIssue]:
    """
    Detect filters using invalid enum values.
    
    Extracts WHERE clause filters and checks if literals match sample_values
    for columns marked as safe_for_enum_constraints.
    
    Args:
        repair_input: Repair context
        
    Returns:
        List of RepairIssue for each enum mismatch found
    """
    issues: List[RepairIssue] = []
    
    # Extract filters from WHERE clause
    filters = _extract_where_filters(repair_input.original_sql)
    schema_context = repair_input.schema_context
    
    for column_name, raw_literal, value_clean in filters:
        # Find column metadata
        column_info = _find_column(schema_context, column_name)
        if column_info is None:
            continue
        
        table_name, column = column_info
        
        # Get sample values and profile
        sample_values = _get_sample_values(column)
        profile = _get_column_profile(column)
        
        # Skip if not a valid enum column
        if (
            not sample_values
            or len(sample_values) > repair_input.config.max_enum_values_per_column
            or not getattr(profile, "safe_for_enum_constraints", False)
        ):
            continue
        
        # Check if value is in sample_values (case-insensitive)
        normalized_samples = {str(val).lower() for val in sample_values}
        if value_clean.lower() in normalized_samples:
            continue
        
        # Found a mismatch
        issues.append(
            RepairIssue(
                issue_type="enum_value_mismatch",
                column=getattr(column, "name", column_name),
                table=table_name,
                value_used=value_clean,
                suggested_values=list(sample_values),
                details={
                    "raw_literal": raw_literal,
                    "sample_values": sample_values,
                    "profile_semantic_role": getattr(profile, "semantic_role", "unknown"),
                },
            )
        )
    
    return issues


def _extract_question_year(question: str) -> Optional[int]:
    """Extract a 4-digit year from the question."""
    match = YEAR_PATTERN.search(question)
    return int(match.group(0)) if match else None


def _extract_where_filters(sql: str) -> List[Tuple[str, str, str]]:
    """
    Parse WHERE clause and extract (column_name, raw_literal, clean_value) triples.
    
    Args:
        sql: SQL query
        
    Returns:
        List of (column_name, raw_literal, clean_value) tuples
    """
    lower_sql = sql.lower()
    where_match = WHERE_SPLIT_PATTERN.split(lower_sql, maxsplit=1)
    if len(where_match) < 2:
        return []
    
    # Get the part after WHERE from original SQL (preserving case)
    original_after_where = sql[len(sql) - len(where_match[1]) :]
    
    # Find the boundary (GROUP BY, ORDER BY, LIMIT)
    boundary_match = CLAUSE_BOUNDARY_PATTERN.search(original_after_where)
    target_clause = (
        original_after_where[: boundary_match.start()]
        if boundary_match
        else original_after_where
    )
    
    # Extract filters
    filters: List[Tuple[str, str, str]] = []
    for line in re.split(r'\bAND\b', target_clause, flags=re.IGNORECASE):
        match = FILTER_PATTERN.search(line)
        if not match:
            continue
        
        raw_literal = match.group("value").strip()
        value_clean = raw_literal.strip('"\'')
        column_name = match.group("column").strip('"')
        
        filters.append((column_name, raw_literal, value_clean))
    
    return filters


def _find_column(
    schema_context: SchemaContext,
    target_column: str,
) -> Optional[Tuple[str, ColumnMetadata]]:
    """
    Find a column by name in the schema context.
    
    Args:
        schema_context: Schema metadata
        target_column: Column name to find
        
    Returns:
        Tuple of (table_name, ColumnMetadata) if found, None otherwise
    """
    normalized = target_column.lower()
    for table in getattr(schema_context.selected_schema, "tables", []):
        for column in getattr(table, "columns", []):
            column_name = getattr(column, "name", getattr(column, "column_name", "")).lower()
            if column_name == normalized:
                return table.table_name, column
    return None


def _get_sample_values(column: ColumnMetadata) -> List[str]:
    """Extract sample_values from column metadata."""
    if hasattr(column, "sample_values") and column.sample_values:
        return list(column.sample_values)
    extra = getattr(column, "extra", {})
    return list(extra.get("sample_values", []))


def _get_column_profile(column: ColumnMetadata):
    """Extract profile from column metadata."""
    if hasattr(column, "profile") and column.profile is not None:
        return column.profile
    extra = getattr(column, "extra", {})
    return extra.get("profile", None)
