from __future__ import annotations

from typing import Any, Dict

from ..exec_sql.validation import validate_for_execution
from .types import RepairInput, RepairPatch


def evaluate_patch(
    repair_input: RepairInput,
    patch: RepairPatch,
) -> Any:
    """
    Validate and execute the patched SQL using the configured runner.
    """
    validation_kwargs: Dict[str, Any] = {
        "sql": patch.new_sql,
        "engine": repair_input.engine,
        "expected_shape": repair_input.expected_shape,
    }
    validation = validate_for_execution(**validation_kwargs)
    if not validation["safe_to_execute"]:
        raise ValueError(f"Patched SQL failed safety validation: {validation['errors']}")

    exec_runner = repair_input.config.exec_runner
    if exec_runner is None:
        raise RuntimeError("RepairConfig.exec_runner must be provided to evaluate patches.")

    return exec_runner(patch.new_sql)