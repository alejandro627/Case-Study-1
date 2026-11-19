from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from . import config
from .analysis import evaluate_umda, uv_sensitivity
from .exporters import write_csv
from .scenarios import EVENT_UV_MAP, SCENARIOS
from .uncertainty import UV_ORDER, lhs_unit

# Reference design for the disruption sensitivity analysis: the default design of the deterministic model with v_conv
# = 2 (v_conv = 1 limits the steady-state throughput to 70 % of nominal, below the TST/TUT recovery threshold, which
# would saturate these two metrics in every scenario).
REFERENCE_DESIGN: Dict[str, int] = {
    "a_CNC": 2, "a_A": 3, "a_QC": 2, "r_A": 2, "r_QC": 3,
    "v_conv": 2, "u_sched": 4, "u_maint": 4, "u_inspect": 2, "u_energy": 1,
}

# "One by one" (each disruption alone) then "all together" (all three at once).
TARGET_SCENARIO_NAMES = [
    "CNC_BREAKDOWN_ONLY",
    "COBOT_BREAKDOWN_ONLY",
    "ASSEMBLY_ERROR_ONLY",
    "ALL_THREE_DISRUPTIONS",
]

ACTIVE_THRESHOLD = 0.05  # |SRC| below this is treated as "no measurable effect"

def _mode_by_name(name: str) -> int:
    for mode, scenario in SCENARIOS.items():
        if scenario["name"] == name:
            return mode
    raise KeyError(f"Unknown scenario name: {name}")

def _expected_uv(scenario: Mapping[str, Any]) -> Optional[str]:
    events = scenario["events"]
    if len(events) != 1:
        return None
    event_type = events[0][0]
    return EVENT_UV_MAP[event_type]

def _check_scenario(name: str, expected_uv: Optional[str], sens_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [f"[{name}]"]
    if expected_uv is not None:
        by_output: Dict[str, List[Mapping[str, Any]]] = {}
        for r in sens_rows:
            by_output.setdefault(r["output"], []).append(r)
        for out, rows in by_output.items():
            top = max(rows, key=lambda r: r["absolute_SRC"])
            if top["absolute_SRC"] < ACTIVE_THRESHOLD:
                match = "NO EFFECT (expected uv shows no measurable sensitivity)"
            elif top["uncertain_variable"] == expected_uv:
                match = "OK"
            else:
                match = "MISMATCH"
            lines.append(
                f"  output={out:22s} top_uv={top['uncertain_variable']:16s} "
                f"|SRC|={top['absolute_SRC']:.3f} expected_active_uv={expected_uv} -> {match}"
            )
        for uv in UV_ORDER:
            if uv == expected_uv:
                continue
            max_abs = max((r["absolute_SRC"] for r in sens_rows if r["uncertain_variable"] == uv), default=0.0)
            status = "OK (inactive, as expected)" if max_abs < ACTIVE_THRESHOLD else "UNEXPECTED COUPLING"
            lines.append(f"  inactive-check {uv:16s} max|SRC|={max_abs:.3f} -> {status}")
    else:
        for uv in UV_ORDER:
            max_abs = max((r["absolute_SRC"] for r in sens_rows if r["uncertain_variable"] == uv), default=0.0)
            status = "OK (active, as expected)" if max_abs >= ACTIVE_THRESHOLD else "WEAK/NO EFFECT"
            lines.append(f"  active-check {uv:16s} max|SRC|={max_abs:.3f} -> {status}")
    return "\n".join(lines)

def run(design: Optional[Mapping[str, int]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    design = dict(design) if design is not None else dict(REFERENCE_DESIGN)
    lhs = lhs_unit(config.N_LHS_SENSITIVITY, len(UV_ORDER), config.LHS_SEED + 9000)

    original_mode = config.SCENARIO_MODE
    sensitivity_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    report_lines: List[str] = []
    try:
        for name in TARGET_SCENARIO_NAMES:
            mode = _mode_by_name(name)
            scenario = SCENARIOS[mode]
            config.SCENARIO_MODE = mode

            summary, runs, _ = evaluate_umda(design, lhs, keep_profiles=False)
            sens = uv_sensitivity(runs)
            expected_uv = _expected_uv(scenario)

            for r in sens:
                sensitivity_rows.append({
                    "scenario_mode": mode, "scenario_name": name,
                    "expected_active_uv": expected_uv, **r,
                })
            summary_rows.append({"scenario_mode": mode, "scenario_name": name, **summary})
            report_lines.append(_check_scenario(name, expected_uv, sens))
    finally:
        config.SCENARIO_MODE = original_mode

    return sensitivity_rows, summary_rows, report_lines

def export(sensitivity_rows: Sequence[Mapping[str, Any]], summary_rows: Sequence[Mapping[str, Any]]) -> None:
    out = config.OUTPUT_DIR
    write_csv(out / "disruption_sensitivity_lhs_individual_and_combined.csv", sensitivity_rows)
    write_csv(out / "disruption_sensitivity_lhs_summary.csv", summary_rows)
