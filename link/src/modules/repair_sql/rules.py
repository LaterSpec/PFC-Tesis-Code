"""
Rule-based heuristics for generating SQL patches.

Each rule takes detected issues and produces a RepairPatch if applicable.
"""
from __future__ import annotations

import re
from typing import List, Optional

from . import mappers
from .types import RepairInput, RepairIssue, RepairPatch

CLAUSE_BOUNDARY_PATTERN = re.compile(r"\b(GROUP\s+BY|ORDER\s+BY|LIMIT)\b", re.IGNORECASE)


def apply_year_repair(
    repair_input: RepairInput,
    issues: List[RepairIssue],
) -> Optional[RepairPatch]:
    """
    Add a missing year filter to the WHERE clause.
    
    Args:
        repair_input: Repair context
        issues: List of detected issues
        
    Returns:
        RepairPatch if a year filter was added, None otherwise
    """
    # Find year issue
    year_issue = next(
        (issue for issue in issues if issue.issue_type == "missing_year_filter"), 
        None
    )
    if year_issue is None:
        return None
    
    # Extract year value
    year_value = year_issue.question_value or year_issue.details.get("year")
    if year_value is None:
        return None
    
    # Build condition
    condition = f'"{year_issue.column or "year"}" = {int(year_value)}'
    
    # Inject condition into SQL
    new_sql = _inject_condition(repair_input.original_sql, condition)
    
    return RepairPatch(
        description=f'Add year filter {condition}',
        new_sql=new_sql,
        issues_resolved=[year_issue],
        metadata={"rule": "year_filter"},
    )


def apply_enum_repairs(
    repair_input: RepairInput,
    issues: List[RepairIssue],
) -> Optional[RepairPatch]:
    """
    Replace invalid enum literals with valid values.
    
    Args:
        repair_input: Repair context
        issues: List of detected issues
        
    Returns:
        RepairPatch if any enum values were mapped, None otherwise
    """
    # Filter enum issues
    enum_issues = [issue for issue in issues if issue.issue_type == "enum_value_mismatch"]
    if not enum_issues:
        return None
    
    new_sql = repair_input.original_sql
    resolved: List[RepairIssue] = []
    mechanisms: List[str] = []
    
    # Try to map each enum issue
    for issue in enum_issues:
        sample_values = issue.details.get("sample_values", [])
        
        # Find the column metadata
        column = _resolve_column_metadata(repair_input, issue.column)
        
        # Try to map the literal
        mapping = mappers.map_literal_to_enum(
            question=repair_input.question,
            column=column,
            literal_value=issue.value_used or "",
            sample_values=sample_values,
            config=repair_input.config,
        )
        
        if not mapping:
            continue
        
        # Build replacement literal with same quote style
        raw_literal = issue.details.get("raw_literal", f"'{issue.value_used}'")
        replacement_literal = _wrap_literal(raw_literal, mapping.value)
        
        # Replace in SQL (only first occurrence)
        updated, new_sql = _replace_literal_once(new_sql, raw_literal, replacement_literal)
        if updated:
            resolved.append(issue)
            mechanisms.append(mapping.mechanism)
    
    if not resolved:
        return None
    
    return RepairPatch(
        description=f"Map {len(resolved)} enum literal(s) to valid values",
        new_sql=new_sql,
        issues_resolved=resolved,
        metadata={"rule": "enum_mapping", "enum_mapping_mechanisms": mechanisms},
    )


def _inject_condition(sql: str, condition: str) -> str:
    """
    Add a condition to the WHERE clause, or create one if missing.
    
    Args:
        sql: Original SQL
        condition: Condition to add (e.g., '"year" = 2000')
        
    Returns:
        Modified SQL with condition added
    """
    has_where = re.search(r'\bWHERE\b', sql, re.IGNORECASE) is not None
    boundary_match = CLAUSE_BOUNDARY_PATTERN.search(sql)
    
    if has_where:
        # Add to existing WHERE with AND
        insert_pos = boundary_match.start() if boundary_match else len(sql)
        return f"{sql[:insert_pos].rstrip()} AND {condition} {sql[insert_pos:]}"
    else:
        # Create new WHERE clause
        insert_pos = boundary_match.start() if boundary_match else len(sql)
        return f"{sql[:insert_pos].rstrip()} WHERE {condition} {sql[insert_pos:]}"


def _replace_literal_once(sql: str, original: str, replacement: str):
    """
    Replace the first occurrence of a literal in SQL.
    
    Args:
        sql: SQL query
        original: Literal to replace
        replacement: New literal
        
    Returns:
        Tuple of (success: bool, new_sql: str)
    """
    if original not in sql:
        return False, sql
    return True, sql.replace(original, replacement, 1)


def _wrap_literal(original_literal: str, new_value: str) -> str:
    """
    Wrap a value with the same quote style as the original.
    
    Args:
        original_literal: Original literal (e.g., "'California'" or '"California"')
        new_value: New value (e.g., "CA")
        
    Returns:
        New literal with same quote style (e.g., "'CA'" or '"CA"')
    """
    quote = "'" if original_literal.strip().startswith("'") else '"'
    return f"{quote}{new_value}{quote}"


def _resolve_column_metadata(repair_input: RepairInput, column_name: Optional[str]):
    """
    Find the ColumnMetadata object for a given column name.
    
    Args:
        repair_input: Repair context
        column_name: Name of column to find
        
    Returns:
        ColumnMetadata if found, None otherwise
    """
    if column_name is None:
        return None
    
    target = column_name.lower()
    for table in getattr(repair_input.schema_context.selected_schema, "tables", []):
        for column in getattr(table, "columns", []):
            name = getattr(column, "name", getattr(column, "column_name", "")).lower()
            if name == target:
                return column
    
    return None
