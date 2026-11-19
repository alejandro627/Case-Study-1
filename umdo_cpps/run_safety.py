"""Reproducibility utilities: run-specific output directories, machine-readable configuration snapshots and
immediate saving of the raw NSGA-II result (before the refinement). optimisation.py does not import this module.
"""
from __future__ import annotations
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from . import config
from .design import DV_LEVELS, DV_ORDER, XL, XU, vector_to_design

# Source files whose SHA-256 is recorded in every configuration snapshot, to identify the code that produced a results
# folder.
_SOURCE_FILES_FOR_HASH = [
    "config.py", "optimisation.py", "metrics.py", "analysis.py", "des.py",
    "design.py", "main.py", "uncertainty.py", "scenarios.py",
    "deterministic_adapter.py", "run_safety.py",
]


def make_run_dir(label: str, base: Optional[Path] = None) -> Path:
    """Create a new directory base/<label>_<YYYYMMDD_HHMMSS>[_NN] (NN only on a same-second collision), so that every
        run writes to its own directory.
    """
    base = base or config.OUTPUT_DIR
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / f"{label}_{stamp}"
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = base / f"{label}_{stamp}_{suffix:02d}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _file_sha256_16(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def config_snapshot(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Machine-readable record of every setting that determines the results: NSGA-II operators, LHS sample sizes and
        seeds, scenario, design-variable bounds and levels, objective and constraint definitions, resilience-metric
        parameters, package versions, time stamp and source-file hashes. `extra` adds run-specific keys.
    """
    import matplotlib
    import openpyxl
    import pymoo

    pkg_dir = Path(__file__).resolve().parent
    from .optimisation import _G_KEYS  # local import: avoids a circular dependency

    snapshot: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "software_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pymoo": pymoo.__version__,
            "openpyxl": openpyxl.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "source_file_sha256_16": {
            name: _file_sha256_16(pkg_dir / name) for name in _SOURCE_FILES_FOR_HASH
        },
        "nsga2": {
            "population": config.NSGA2_POPULATION,
            "generations": config.NSGA2_GENERATIONS,
            "seed": config.NSGA2_SEED,
            "population_lhs_seed": config.POPULATION_LHS_SEED,
            "crossover": {"operator": "SBX", "prob": config.NSGA2_CROSSOVER_PROB, "eta": config.NSGA2_CROSSOVER_ETA},
            "mutation": {"operator": "PM", "prob": f"1/{len(DV_ORDER)}", "eta": config.NSGA2_MUTATION_ETA},
            "integer_repair": "RoundingRepair (pymoo.operators.repair.rounding)" if config.NSGA2_USE_INTEGER_REPAIR else None,
            "eliminate_duplicates_epsilon": 1e-16,
            "fail_fast_threshold_exceptions": config.NSGA2_FAIL_FAST_THRESHOLD,
            "checkpoint_every_n_gen": config.NSGA2_CHECKPOINT_EVERY_N_GEN,
        },
        "lhs": {
            "N_LHS_OPTIMISATION": config.N_LHS_OPTIMISATION,
            "N_LHS_FINAL": config.N_LHS_FINAL,
            "N_LHS_SENSITIVITY": config.N_LHS_SENSITIVITY,
            "LHS_SEED": config.LHS_SEED,
        },
        "scenario": {"active_mode": config.SCENARIO_MODE},
        "design_variables": {
            "order": DV_ORDER, "levels": DV_LEVELS,
            "xl": XL.tolist(), "xu": XU.tolist(),
        },
        "objectives": {
            "formula": "F = [-disrupted_throughput_mean, energy_kWh_mean, -R_phi_tf_mean]",
            "directions": {
                "disrupted_throughput_mean": "maximize",
                "energy_kWh_mean": "minimize",
                "R_phi_tf_mean": "maximize",
            },
            "note": "TTR/TST/R_rate are descriptive-only; never an optimization objective.",
        },
        "constraints_g_keys": list(_G_KEYS),
        "resilience_metric_parameters": {
            "RECOVERY_THRESHOLD_FRACTION": config.RECOVERY_THRESHOLD_FRACTION,
            "R_PHI_EVALUATION_MODE": config.R_PHI_EVALUATION_MODE,
            "R_PHI_OFFSET_MIN": config.R_PHI_OFFSET_MIN,
            "RELIABILITY_LEVEL": config.RELIABILITY_LEVEL,
            "EPSILON": config.EPSILON,
            "DT_MIN": config.DT_MIN,
            "SHIFT_HOURS": config.SHIFT_HOURS,
        },
    }
    if extra:
        snapshot.update(extra)
    return snapshot


def write_json(path: Path, data: Dict[str, Any]) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_default)


def write_config_snapshot(out_dir: Path, extra: Optional[Dict[str, Any]] = None) -> Path:
    path = out_dir / "run_config.json"
    write_json(path, config_snapshot(extra))
    return path


def save_nsga2_raw_result(out_dir: Path, res, path_name: str = "nsga2_raw_result.csv") -> Path:
    """Save the final NSGA-II population right after run_nsga2(), before the refinement: one row per individual with
        its (rounded) design variables, objective values, constraint values and feasibility flag (all G <= 0).
    """
    from .exporters import write_csv
    from .optimisation import _G_KEYS

    X = np.atleast_2d(res.X)
    F = np.atleast_2d(res.F)
    G = np.atleast_2d(res.G) if res.G is not None else np.zeros((len(X), len(_G_KEYS)))
    rows = []
    for i in range(len(X)):
        design = vector_to_design(X[i])
        row: Dict[str, Any] = {"solution_id": i + 1, **design}
        row["F_neg_throughput"] = float(F[i, 0])
        row["F_energy"] = float(F[i, 1])
        row["F_neg_R_phi_tf"] = float(F[i, 2])
        for k, gk in enumerate(_G_KEYS):
            row[gk] = float(G[i, k])
        row["feasible_raw"] = bool(np.all(G[i] <= 0))
        rows.append(row)
    path = out_dir / path_name
    write_csv(path, rows)
    return path
