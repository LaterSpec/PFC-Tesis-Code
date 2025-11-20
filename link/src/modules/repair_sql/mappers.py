"""
Mapping logic for converting invalid enum literals to valid values.

Supports two mechanisms:
1. Dictionary-based mapping (fast, deterministic)
2. LLM-based mapping (flexible, requires model)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..inspect_schema.types import ColumnMetadata
from .config import RepairConfig


@dataclass
class EnumMappingResult:
    """
    Result of mapping an invalid literal to a valid enum value.
    
    Attributes:
        value: The valid enum value to use
        mechanism: How the mapping was found ("dict" or "llm")
    """
    value: str
    mechanism: str  # "dict" | "llm"


# Dictionary mappings for common enum columns
STATE_NAME_TO_ABBR = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "district of columbia": "DC",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}

GENDER_MAP = {
    "female": "F",
    "f": "F",
    "woman": "F",
    "male": "M",
    "m": "M",
    "man": "M",
}


def map_literal_to_enum(
    question: str,
    column: ColumnMetadata,
    literal_value: str,
    sample_values: List[str],
    config: RepairConfig,
) -> Optional[EnumMappingResult]:
    """
    Try to map an invalid literal to a valid enum value.
    
    First attempts dictionary mapping, then LLM if enabled.
    
    Args:
        question: Original question (context for LLM)
        column: Column metadata
        literal_value: The invalid value to map
        sample_values: Valid enum values for this column
        config: RepairConfig
        
    Returns:
        EnumMappingResult if mapping found, None otherwise
    """
    # Try dictionary mapping first
    mapping = map_literal_to_enum_with_dict(column, literal_value)
    if mapping:
        return mapping
    
    # Try LLM mapping if enabled
    if config.enable_llm_enum_mapper:
        mapping = map_literal_to_enum_with_llm(
            question=question,
            column=column,
            literal_value=literal_value,
            sample_values=sample_values,
            config=config,
        )
        if mapping:
            return mapping
    
    return None


def map_literal_to_enum_with_dict(
    column: ColumnMetadata,
    literal_value: str,
) -> Optional[EnumMappingResult]:
    """
    Use hardcoded dictionaries to map common enum values.
    
    Args:
        column: Column metadata
        literal_value: Value to map
        
    Returns:
        EnumMappingResult if mapping found, None otherwise
    """
    column_name = getattr(column, "name", getattr(column, "column_name", "")).lower()
    candidate = literal_value.lower()
    
    # State mapping
    if column_name == "state":
        mapped = STATE_NAME_TO_ABBR.get(candidate)
        if mapped:
            return EnumMappingResult(value=mapped, mechanism="dict")
    
    # Gender mapping
    if column_name == "gender":
        mapped = GENDER_MAP.get(candidate)
        if mapped:
            return EnumMappingResult(value=mapped, mechanism="dict")
    
    return None


def map_literal_to_enum_with_llm(
    question: str,
    column: ColumnMetadata,
    literal_value: str,
    sample_values: List[str],
    config: RepairConfig,
) -> Optional[EnumMappingResult]:
    """
    Use an LLM to map an invalid literal to a valid enum value.
    
    Constructs a prompt with the question, column name, current literal,
    and valid values, then asks the LLM to choose the best match.
    
    Args:
        question: Original question
        column: Column metadata
        literal_value: Invalid value to map
        sample_values: Valid enum values
        config: RepairConfig with LLM settings
        
    Returns:
        EnumMappingResult if LLM returns a valid value, None otherwise
    """
    if not config.llm_mapper_pipeline or not sample_values:
        return None
    
    column_name = getattr(column, 'name', getattr(column, 'column_name', 'unknown'))
    
    prompt = (
        "You map literals to valid enum codes.\n"
        f"Question: {question}\n"
        f"Column: {column_name}\n"
        f"Current literal: {literal_value}\n"
        f"Valid values: {sample_values}\n"
        "Respond with exactly one of the valid values or NONE."
    )
    
    generator = config.llm_mapper_pipeline
    try:
        raw = generator(
            prompt,
            max_new_tokens=config.llm_max_new_tokens,
            temperature=config.llm_temperature,
            repetition_penalty=config.llm_repetition_penalty,
            do_sample=False,
        )
        text = raw[0]["generated_text"].strip() if raw else ""
        candidate = text.split()[0].strip('"\'')
        
        # Check if candidate is in sample_values
        if candidate in sample_values:
            return EnumMappingResult(value=candidate, mechanism="llm")
    except Exception as e:
        # Log error but don't crash
        print(f"LLM mapping failed: {e}")
    
    return None
