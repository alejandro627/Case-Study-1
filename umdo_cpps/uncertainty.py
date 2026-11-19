from __future__ import annotations
from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, Mapping
import numpy as np
from . import config
from .design import DV_ORDER

@dataclass(frozen=True)
class Distribution:
    mean: float
    std: float
    unit: str = "min"

# Uncertain durations of the three disruptions: TTR_CNC (CNC breakdown), TTR_Cobot (cobot breakdown) and
# Tcorr_Assembly (assembly operator error). Sampling is not truncated (see normal_quantile). After the recovery ramp
# the capacity returns to nominal (des.py); recoverability follows from the simulation (buffers, ramps, cartons
# produced).
BASELINE_UV: Dict[str, Distribution] = {
    "TTR_CNC": Distribution(45.0, 15.0),
    "TTR_Cobot": Distribution(35.0, 12.0),
    "Tcorr_Assembly": Distribution(18.0, 6.0),
}
UV_ORDER = list(BASELINE_UV)

COEFFICIENTS: Dict[str, Dict[int, Dict[str, float]]] = {
    "a_CNC": {1: {}, 2: {}, 3: {}},
    "a_A": {1: {"Tcorr_Assembly": 1.00}, 2: {"Tcorr_Assembly": 0.87}, 3: {"Tcorr_Assembly": 0.76}},
    "a_QC": {1: {}, 2: {}, 3: {}},
    "r_A": {1: {"Tcorr_Assembly": 1.00}, 2: {"Tcorr_Assembly": 0.92}, 3: {"Tcorr_Assembly": 0.78}},
    "r_QC": {1: {}, 2: {}, 3: {}},
    "v_conv": {1: {}, 2: {}, 3: {}, 4: {}},
    "u_sched": {
        1: {"TTR_CNC": 1.00, "TTR_Cobot": 1.00},
        2: {"TTR_CNC": 0.99, "TTR_Cobot": 0.99},
        3: {"TTR_CNC": 0.97, "TTR_Cobot": 0.97},
        4: {"TTR_CNC": 0.94, "TTR_Cobot": 0.94},
        5: {"TTR_CNC": 0.90, "TTR_Cobot": 0.90},
    },
    "u_maint": {
        1: {"TTR_CNC":1,"TTR_Cobot":1,"Tcorr_Assembly":1},
        2: {"TTR_CNC":.94,"TTR_Cobot":.94,"Tcorr_Assembly":.99},
        3: {"TTR_CNC":.86,"TTR_Cobot":.86,"Tcorr_Assembly":.98},
        4: {"TTR_CNC":.76,"TTR_Cobot":.76,"Tcorr_Assembly":.96},
        5: {"TTR_CNC":.65,"TTR_Cobot":.65,"Tcorr_Assembly":.93},
    },
    "u_inspect": {
        1: {"TTR_CNC":1,"TTR_Cobot":1,"Tcorr_Assembly":1},
        2: {"TTR_CNC":.96,"TTR_Cobot":.95,"Tcorr_Assembly":.92},
        3: {"TTR_CNC":.90,"TTR_Cobot":.88,"Tcorr_Assembly":.86},
    },
    "u_energy": {1: {}, 2: {}, 3: {}, 4: {}},
}

def combined_mean_coefficient(design: Mapping[str, int], variable: str) -> float:
    value = 1.0
    for dv in DV_ORDER:
        value *= COEFFICIENTS[dv][int(design[dv])].get(variable, 1.0)
    return value

def std_coefficient(k_mean: float) -> float:
    if config.STD_MODE == "fixed":
        return 1.0
    if config.STD_MODE == "half_effect":
        return 1.0 - 0.5 * (1.0 - k_mean)
    if config.STD_MODE == "same_coefficient":
        return k_mean
    raise ValueError(f"Unknown STD_MODE: {config.STD_MODE}")

def distributions_for_design(design: Mapping[str, int]) -> Dict[str, Distribution]:
    result = {}
    for variable, baseline in BASELINE_UV.items():
        km = combined_mean_coefficient(design, variable)
        ks = std_coefficient(km)
        mean = baseline.mean * km
        std = max(1e-6, baseline.std * ks)
        result[variable] = Distribution(mean, std, baseline.unit)
    return result

def lhs_unit(n: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = np.empty((n, dimensions), dtype=float)
    for j in range(dimensions):
        values = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(values)
        matrix[:, j] = values
    return matrix

def normal_quantile(u: float, d: Distribution) -> float:
    normal = NormalDist(d.mean, d.std)
    p = max(1e-12, min(1 - 1e-12, u))
    return normal.inv_cdf(p)

def samples_for_design(design: Mapping[str, int], unit_lhs: np.ndarray):
    distributions = distributions_for_design(design)
    return [
        {uv: normal_quantile(float(row[i]), distributions[uv]) for i, uv in enumerate(UV_ORDER)}
        for row in unit_lhs
    ]
