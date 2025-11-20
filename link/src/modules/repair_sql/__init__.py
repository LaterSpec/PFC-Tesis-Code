"""Repair module for post-execution SQL fixes."""
from __future__ import annotations

from typing import Dict, Optional

from ..exec_sql.types import ExecutionResult
from ..inspect_schema.types import SchemaContext
from . import detection, patching, rules, signals
from .config import DEFAULT_REPAIR_CONFIG, RepairConfig
from .types import RepairInput, RepairResult


def should_trigger_repair(
    gen_signals: Dict[str, float],
    exec_signals: Dict[str, float],
) -> bool:
    """
    Decide whether repairing is worth attempting based on SCM signals.
    """
    rows_empty = exec_signals.get("rows_empty", 0.0)
    exec_error = exec_signals.get("exec_error", 0.0)
    format_ok = gen_signals.get("primary_format_ok", 0.0)
    risk_score = gen_signals.get("risk_score", 1.0)

    return rows_empty == 1.0 and exec_error == 0.0 and format_ok == 1.0 and risk_score <= 0.6


def repair_sql(
    question: str,
    original_sql: str,
    expected_shape: Dict[str, any],
    schema_context: SchemaContext,
    gen_signals: Dict[str, float],
    exec_result: ExecutionResult,
    exec_signals: Dict[str, float],
    engine: str,
    db_config: Optional[Dict[str, any]] = None,
    config: Optional[RepairConfig] = None,
) -> RepairResult:
    repair_config = config or DEFAULT_REPAIR_CONFIG

    repair_input = RepairInput(
        question=question,
        original_sql=original_sql,
        expected_shape=expected_shape,
        schema_context=schema_context,
        gen_signals=gen_signals,
        exec_result=exec_result,
        exec_signals=exec_signals,
        engine=engine,
        db_config=db_config or {},
        config=repair_config,
    )

    detected_issues = detection.detect_issues(repair_input)
    debug_info = {"num_issues": len(detected_issues)}

    patch = None
    if repair_config.enable_year_repairs:
        patch = rules.apply_year_repair(repair_input, detected_issues)

    if patch is None and repair_config.enable_enum_repairs:
        patch = rules.apply_enum_repairs(repair_input, detected_issues)

    repaired_exec = None
    if patch:
        try:
            repaired_exec = patching.evaluate_patch(repair_input, patch)
        except Exception as exc:
            debug_info["reexecute_error"] = str(exc)

    repair_signals = signals.build_repair_signals(
        original_exec=exec_result,
        repaired_exec=repaired_exec,
        issues=detected_issues,
        patch=patch,
    )

    applied = patch is not None and repaired_exec is not None

    return RepairResult(
        applied=applied,
        original_sql=original_sql,
        repaired_sql=patch.new_sql if repaired_exec else None,
        original_exec_result=exec_result,
        repaired_exec_result=repaired_exec,
        issues=detected_issues,
        patch=patch,
        repair_signals=repair_signals.values,
        debug_info=debug_info,
    )