"""Scenario-wise resilience comparison of one fixed design over the disruption scenarios, using evaluate_umda and
the LHS samples. config.SCENARIO_MODE is set per scenario and restored afterwards.
"""
from __future__ import annotations
from typing import Any, Dict, List, Mapping, Sequence
from . import config
from .analysis import evaluate_umda
from .scenarios import get_scenario
from .uncertainty import UV_ORDER, lhs_unit

def run(design: Mapping[str, int], modes: Sequence[int] = (1, 2, 3, 4, 5, 6, 7), n: int = config.N_LHS_SENSITIVITY) -> List[Dict[str, Any]]:
    original_mode = config.SCENARIO_MODE
    rows: List[Dict[str, Any]] = []
    try:
        for mode in modes:
            config.SCENARIO_MODE = mode
            scenario = get_scenario(mode)
            unit_lhs = lhs_unit(n, len(UV_ORDER), config.LHS_SEED + 9000 + mode)
            summary, _, _ = evaluate_umda(design, unit_lhs, keep_profiles=False)
            rows.append({"scenario_mode": mode, "scenario_name": scenario["name"], **summary})
    finally:
        config.SCENARIO_MODE = original_mode
    return rows
