from __future__ import annotations

from typing import List, Optional

from ..exec_sql.types import ExecutionResult
from .types import RepairIssue, RepairPatch, RepairSignals


def build_repair_signals(
    original_exec: ExecutionResult,
    repaired_exec: Optional[ExecutionResult],
    issues: List[RepairIssue],
    patch: Optional[RepairPatch],
) -> RepairSignals:
    values = {
        "repair_applied": 1.0 if patch else 0.0,
        "repair_success": 0.0,
        "repair_row_count_delta": 0.0,
        "repair_exec_latency_delta_ms": 0.0,
        "repair_used_year_rule": 0.0,
        "repair_used_enum_rule": 0.0,
        "repair_used_llm_mapping": 0.0,
    }

    if patch:
        rule = patch.metadata.get("rule")
        if rule == "year_filter":
            values["repair_used_year_rule"] = 1.0
        if rule == "enum_mapping":
            values["repair_used_enum_rule"] = 1.0
            mechanisms = patch.metadata.get("enum_mapping_mechanisms", [])
            if any(mech == "llm" for mech in mechanisms):
                values["repair_used_llm_mapping"] = 1.0

    if repaired_exec:
        row_delta = float((repaired_exec.row_count or 0) - (original_exec.row_count or 0))
        values["repair_row_count_delta"] = row_delta
        values["repair_exec_latency_delta_ms"] = float(
            (repaired_exec.latency_ms or 0.0) - (original_exec.latency_ms or 0.0)
        )
        success = (
            (original_exec.row_count or 0) == 0 and (repaired_exec.row_count or 0) > 0
        ) or (row_delta > 0)
        values["repair_success"] = 1.0 if success else 0.0

    return RepairSignals(values=values)