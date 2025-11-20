"""Configuration dataclass for repair_sql module."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.modules.exec_sql.types import ExecutionResult

ExecRunner = Callable[[str], Any]


@dataclass
class RepairConfig:
    """
    Tunable parameters for the repair_sql module.
    
    Attributes:
        enable_enum_repairs: Whether to apply enum value mapping repairs
        enable_year_repairs: Whether to add missing year filters
        enable_llm_enum_mapper: Whether to use LLM for enum value mapping
        max_enum_values_per_column: Max distinct values to treat column as enum
        llm_mapper_pipeline: HuggingFace pipeline or similar for LLM mapping
        llm_mapper_tokenizer: Tokenizer for the LLM mapper (optional)
        llm_max_new_tokens: Max tokens to generate in LLM mapper
        llm_temperature: Temperature for LLM mapper
        llm_repetition_penalty: Repetition penalty for LLM mapper
        exec_runner: Function that executes SQL and returns ExecutionResult
        exec_validation_kwargs: Additional kwargs for validation
    """
    enable_enum_repairs: bool = True
    enable_year_repairs: bool = True
    enable_llm_enum_mapper: bool = False
    max_enum_values_per_column: int = 100
    llm_mapper_pipeline: Optional[Any] = None
    llm_mapper_tokenizer: Optional[Any] = None
    llm_max_new_tokens: int = 64
    llm_temperature: float = 0.0
    llm_repetition_penalty: float = 1.0
    exec_runner: Optional[ExecRunner] = None
    exec_validation_kwargs: Dict[str, Any] = field(default_factory=dict)


DEFAULT_REPAIR_CONFIG = RepairConfig()
