from __future__ import annotations
import importlib.util, sys
from typing import Any, Mapping, Dict
from . import config

def load_model() -> Any:
    if not config.MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing deterministic model: {config.MODEL_FILE}\n"
            "Copy deterministic_des_mdo_mda_mdf_assistance_no_buffers.py "
            "to the project root."
        )
    spec = importlib.util.spec_from_file_location("deterministic_model", config.MODEL_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {config.MODEL_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["deterministic_model"] = module
    spec.loader.exec_module(module)
    return module

_MODEL = None

def evaluate_mdo(design: Mapping[str, int]) -> Dict[str, float]:
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    vector = _MODEL.DesignVector(**dict(design))
    row = _MODEL.evaluate_mdf(vector)
    return {
        "nominal_throughput": float(row["throughput_cartons_h"]),
        "nominal_energy_kWh": float(row["E_tot_kWh"]),
        "cycle_time_s": float(row["cycle_time_s_per_carton"]),
        "mda_residual": float(row["MDA_final_residual"]),
        "mda_converged": float(row["MDA_converged"]),
        "nominal_WIP": float(row["avg_WIP_units"]),
        "nominal_energy_per_carton": float(row["e_carton_kWh_per_carton"]),
    }

def thresholds() -> Dict[str, float]:
    # Engineering-constraint thresholds, read from the deterministic model module.
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return {
        "THROUGHPUT_MIN": float(_MODEL.THROUGHPUT_MIN),
        "WIP_MAX": float(_MODEL.WIP_MAX),
        "CT_MAX": float(_MODEL.CT_MAX),
        "E_TOT_MAX": float(_MODEL.E_TOT_MAX),
        "E_CARTON_MAX": float(_MODEL.E_CARTON_MAX),
        "MDA_TOL": float(_MODEL.MDA_TOL),
    }
