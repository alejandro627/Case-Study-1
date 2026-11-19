from __future__ import annotations
from typing import Dict, List, Mapping, Sequence
import numpy as np

DV_ORDER = [
    "a_CNC", "a_A", "a_QC", "r_A", "r_QC",
    "v_conv", "u_sched", "u_maint", "u_inspect", "u_energy",
]

DV_LEVELS: Dict[str, List[int]] = {
    "a_CNC": [1, 2, 3],
    "a_A": [1, 2, 3],
    "a_QC": [1, 2, 3],
    "r_A": [1, 2, 3],
    "r_QC": [1, 2, 3],
    "v_conv": [1, 2, 3, 4],
    "u_sched": [1, 2, 3, 4, 5],
    "u_maint": [1, 2, 3, 4, 5],
    "u_inspect": [1, 2, 3],
    "u_energy": [1, 2, 3, 4],
}

XL = np.array([min(DV_LEVELS[dv]) for dv in DV_ORDER], dtype=int)
XU = np.array([max(DV_LEVELS[dv]) for dv in DV_ORDER], dtype=int)

MIN_ENERGY_REFERENCE = {
    "a_CNC": 1, "a_A": 1, "a_QC": 1,
    "r_A": 2, "r_QC": 1, "v_conv": 4,
    "u_sched": 1, "u_maint": 1,
    "u_inspect": 1, "u_energy": 1,
}

def vector_to_design(x: Sequence[float]) -> Dict[str, int]:
    design = {}
    for i, dv in enumerate(DV_ORDER):
        level = int(round(float(x[i])))
        design[dv] = max(min(DV_LEVELS[dv]), min(max(DV_LEVELS[dv]), level))
    return design

def design_key(design: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(int(design[dv]) for dv in DV_ORDER)
