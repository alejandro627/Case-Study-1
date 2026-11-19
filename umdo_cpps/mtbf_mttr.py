"""Empirical MTBF/MTTR for the disruption scenarios (default: the multi-event scenarios 4, 5, 6 and 7).

- MTBF needs at least two observed inter-failure intervals. Every scenario schedules each event type at most once per
  8-h shift, so MTBF cannot be estimated from these scenarios; this is reported explicitly.
- MTTR (mean repair duration) of the CNC and the cobot is estimated from the sampled repair durations (TTR_CNC,
  TTR_Cobot) over the N LHS realisations, separately from the parameters of the input distributions (BASELINE_UV).
- Tcorr_Assembly is an operator correction time, not a machine repair, and is reported in its own metric category.
- A sampled repair that extends beyond the end of the shift is flagged as horizon-censored and excluded from the
  empirical mean (reported as a separate count).
- The model has no MTBF input: BASELINE_UV specifies repair-duration distributions, and the event times are fixed per
  scenario.
"""
from __future__ import annotations
from typing import Any, Dict, List, Mapping, Sequence
import numpy as np
from . import config
from .scenarios import EVENT_UV_MAP, get_scenario
from .uncertainty import BASELINE_UV, UV_ORDER, combined_mean_coefficient, lhs_unit, samples_for_design

MACHINE_REPAIR_UVS = {"TTR_CNC": "CNC", "TTR_Cobot": "Cobot (Assembly)"}
OPERATOR_CORRECTION_UVS = {"Tcorr_Assembly": "Assembly operator"}

def analyze_scenario(mode: int, design: Mapping[str, int], n: int = config.N_LHS_SENSITIVITY, seed: int = config.LHS_SEED + 5000) -> List[Dict[str, Any]]:
    scenario = get_scenario(mode)
    event_types = [e[0] for e in scenario["events"]]
    event_start_h = {et: h for et, h in scenario["events"]}
    uv_present = {EVENT_UV_MAP[et] for et in event_types}
    unit_lhs = lhs_unit(n, len(UV_ORDER), seed)
    samples = samples_for_design(design, unit_lhs)

    rows: List[Dict[str, Any]] = []
    for uv, role in {**MACHINE_REPAIR_UVS, **OPERATOR_CORRECTION_UVS}.items():
        if uv not in uv_present:
            continue
        et = next(e for e, u in EVENT_UV_MAP.items() if u == uv and e in event_types)
        event_start_min = event_start_h[et] * 60.0
        values = np.asarray([float(s[uv]) for s in samples])

        horizon_censored = values > (config.SHIFT_HOURS * 60.0 - event_start_min)
        n_horizon_censored = int(np.sum(horizon_censored))
        used = values[~horizon_censored]

        n_occurrences_per_shift = sum(1 for e in event_types if e == et)  # always 1 for every current scenario definition

        rows.append({
            "scenario_mode": mode,
            "scenario_name": scenario["name"],
            "machine_or_role": role,
            "metric_category": "machine_repair_MTTR" if uv in MACHINE_REPAIR_UVS else "operator_correction_time_NOT_MTTR",
            "event_type": et,
            "uncertain_variable": uv,
            "n_occurrences_per_shift_in_scenario_definition": n_occurrences_per_shift,
            "MTBF_estimable": n_occurrences_per_shift >= 2,
            "MTBF_min": None if n_occurrences_per_shift >= 2 else "not estimable (scenario schedules only one occurrence of this event per shift - no inter-failure interval exists)",
            "n_LHS_realizations": n,
            "n_horizon_censored_repairs": n_horizon_censored,
            "empirical_mean_duration_min": float(np.mean(used)) if used.size else float("nan"),
            "empirical_std_duration_min": float(np.std(used, ddof=1)) if used.size > 1 else 0.0,
            "empirical_q05_min": float(np.percentile(used, 5)) if used.size else float("nan"),
            "empirical_q95_min": float(np.percentile(used, 95)) if used.size else float("nan"),
            "model_input_mean_min": BASELINE_UV[uv].mean * combined_mean_coefficient(design, uv),
            "model_input_std_min_baseline": BASELINE_UV[uv].std,
            "model_input_note": "distribution PARAMETER used to generate samples above, NOT an empirical estimate",
        })
    return rows

def run(design: Mapping[str, int], modes: Sequence[int] = (4, 5, 6, 7)) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for mode in modes:
        all_rows.extend(analyze_scenario(mode, design))
    return all_rows
