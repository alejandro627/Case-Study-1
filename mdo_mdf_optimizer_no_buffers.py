"""
MDO/MDF optimization for the wooden-box CPPS using the validated deterministic
DES + MDA/MDF model with assistance-oriented design variables.

Required file in the same folder:
    deterministic_des_mdo_mda_mdf_assistance_no_buffers.py

Objectives:
    1) Maximize throughput [cartons/h]
    2) Minimize total energy [kWh/shift]

Exports:
    - mdo_mdf_all_designs.csv
    - mdo_mdf_feasible_designs.csv
    - mdo_mdf_pareto_front.csv
    - mdo_mdf_selected_representative_designs.csv
"""

from __future__ import annotations

import csv
import importlib.util
import itertools
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


# =============================================================================
# SETTINGS
# =============================================================================
MODEL_FILE = Path("deterministic_des_mdo_mda_mdf_assistance_no_buffers.py")
OUTPUT_DIR = Path(".")

# "exhaustive" evaluates all possible combinations.
# "random" evaluates N_RANDOM_DESIGNS random combinations.
OPTIMIZATION_MODE = "exhaustive"

N_RANDOM_DESIGNS = 5000
RANDOM_SEED = 42

# Constraint thresholds of the optimisation.
USE_TIGHTENED_CONSTRAINTS = True

THROUGHPUT_MIN = 1.30       # cartons/h
WIP_MAX = 25.0              # units
CT_MAX = 3000.0             # s/carton
E_TOT_MAX = 27.0            # kWh/shift
E_CARTON_MAX = 2.50         # kWh/carton


# =============================================================================
# DYNAMIC IMPORT
# =============================================================================
def load_module(module_name: str, file_path: Path):
    """
    Dynamically load the deterministic MDA/MDF model.

    sys.modules registration is needed for Python 3.12 dataclass compatibility.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find {file_path}. Place this script in the same folder "
            f"as deterministic_des_mdo_mda_mdf_assistance_no_buffers.py."
        )

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# =============================================================================
# DESIGN GENERATION
# =============================================================================
def generate_all_designs(model) -> Iterable[Any]:
    names = list(model.DESIGN_SPACE.keys())
    values = [model.DESIGN_SPACE[name] for name in names]

    for combo in itertools.product(*values):
        z_dict = dict(zip(names, combo))
        yield model.DesignVector(**z_dict)


def generate_random_designs(model, n: int, seed: int) -> Iterable[Any]:
    rng = random.Random(seed)
    names = list(model.DESIGN_SPACE.keys())

    for _ in range(n):
        z_dict = {name: rng.choice(model.DESIGN_SPACE[name]) for name in names}
        yield model.DesignVector(**z_dict)


# =============================================================================
# CONSTRAINT HANDLING
# =============================================================================
def apply_tightened_constraints(model) -> None:
    model.THROUGHPUT_MIN = THROUGHPUT_MIN
    model.WIP_MAX = WIP_MAX
    model.CT_MAX = CT_MAX
    model.E_TOT_MAX = E_TOT_MAX
    model.E_CARTON_MAX = E_CARTON_MAX


def total_constraint_violation(row: Dict[str, Any]) -> float:
    keys = [
        "viol_TH_min",
        "viol_WIP_max",
        "viol_CT_max",
        "viol_E_tot_max",
        "viol_e_carton_max",
        "viol_MDA_convergence",
    ]
    return sum(float(row.get(k, 0.0)) for k in keys)


def is_feasible(row: Dict[str, Any], tol: float = 1e-9) -> bool:
    return total_constraint_violation(row) <= tol


# =============================================================================
# PARETO FRONT
# =============================================================================
def dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """
    a dominates b if:
    - a has equal or higher throughput
    - a has equal or lower energy
    - a is strictly better in at least one objective
    """
    a_th = float(a["throughput_cartons_h"])
    b_th = float(b["throughput_cartons_h"])

    a_e = float(a["E_tot_kWh"])
    b_e = float(b["E_tot_kWh"])

    return (
        a_th >= b_th
        and a_e <= b_e
        and (a_th > b_th or a_e < b_e)
    )


def pareto_front(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    front: List[Dict[str, Any]] = []

    for i, candidate in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            if dominates(other, candidate):
                dominated = True
                break

        if not dominated:
            front.append(candidate)

    front.sort(key=lambda r: (float(r["E_tot_kWh"]), -float(r["throughput_cartons_h"])))
    return front


# =============================================================================
# REPRESENTATIVE DESIGN SELECTION
# =============================================================================
def normalize(value: float, low: float, high: float, higher_is_better: bool) -> float:
    if high <= low:
        return 0.0

    x = (value - low) / (high - low)
    x = max(0.0, min(1.0, x))

    if higher_is_better:
        return x

    return 1.0 - x


def design_key(row: Dict[str, Any]) -> tuple:
    keys = [
        "a_CNC", "a_A", "a_QC", "r_A", "r_QC",
        "v_conv", "u_sched", "u_maint", "u_inspect", "u_energy",
    ]
    return tuple((k, row[k]) for k in keys)


def select_representative_designs(front: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not front:
        return []

    best_perf = max(front, key=lambda r: float(r["throughput_cartons_h"]))
    best_energy = min(front, key=lambda r: float(r["E_tot_kWh"]))

    th_values = [float(r["throughput_cartons_h"]) for r in front]
    e_values = [float(r["E_tot_kWh"]) for r in front]

    th_min, th_max = min(th_values), max(th_values)
    e_min, e_max = min(e_values), max(e_values)

    def balanced_score(r: Dict[str, Any]) -> float:
        th_norm = normalize(float(r["throughput_cartons_h"]), th_min, th_max, higher_is_better=True)
        e_norm = normalize(float(r["E_tot_kWh"]), e_min, e_max, higher_is_better=False)
        return 0.5 * th_norm + 0.5 * e_norm

    balanced = max(front, key=balanced_score)

    selected: List[Dict[str, Any]] = []
    seen = set()

    for label, row in [
        ("best_performance", best_perf),
        ("best_energy", best_energy),
        ("balanced", balanced),
    ]:
        key = design_key(row)
        if key not in seen:
            out = dict(row)
            out["selection_label"] = label
            selected.append(out)
            seen.add(key)

    return selected


# =============================================================================
# CSV HELPERS
# =============================================================================
def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_optimization_columns(rows: List[Dict[str, Any]]) -> None:
    for r in rows:
        r["total_constraint_violation"] = total_constraint_violation(r)
        r["feasible"] = is_feasible(r)
        r["performance_objective_max_throughput"] = r["throughput_cartons_h"]
        r["sustainability_objective_min_energy"] = r["E_tot_kWh"]


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================
def main() -> None:
    model = load_module("mda_model_no_buffers", MODEL_FILE)

    if USE_TIGHTENED_CONSTRAINTS:
        apply_tightened_constraints(model)

    print("=" * 100)
    print("MDO/MDF OPTIMIZATION: WOODEN-BOX CPPS")
    print("=" * 100)
    print(f"Model file: {MODEL_FILE.resolve()}")
    print(f"Optimization mode: {OPTIMIZATION_MODE}")
    print(f"Tightened constraints: {USE_TIGHTENED_CONSTRAINTS}")

    if USE_TIGHTENED_CONSTRAINTS:
        print("\nConstraints used:")
        print(f"  THROUGHPUT_MIN = {model.THROUGHPUT_MIN} cartons/h")
        print(f"  WIP_MAX        = {model.WIP_MAX} units")
        print(f"  CT_MAX         = {model.CT_MAX} s/carton")
        print(f"  E_TOT_MAX      = {model.E_TOT_MAX} kWh/shift")
        print(f"  E_CARTON_MAX   = {model.E_CARTON_MAX} kWh/carton")

    if OPTIMIZATION_MODE == "exhaustive":
        designs = list(generate_all_designs(model))
    elif OPTIMIZATION_MODE == "random":
        designs = list(generate_random_designs(model, N_RANDOM_DESIGNS, RANDOM_SEED))
    else:
        raise ValueError("OPTIMIZATION_MODE must be 'exhaustive' or 'random'.")

    print(f"\nNumber of designs to evaluate: {len(designs)}")

    rows: List[Dict[str, Any]] = []

    for idx, z in enumerate(designs, start=1):
        row = model.evaluate_mdf(z)
        row["design_id"] = idx
        rows.append(row)

        if idx % 10000 == 0:
            print(f"  evaluated {idx}/{len(designs)} designs...")

    add_optimization_columns(rows)

    feasible_rows = [r for r in rows if r["feasible"]]
    front = pareto_front(feasible_rows)
    selected = select_representative_designs(front)

    front_keys = {design_key(r) for r in front}

    for r in rows:
        r["pareto_front"] = design_key(r) in front_keys

    out_all = OUTPUT_DIR / "mdo_mdf_all_designs.csv"
    out_feasible = OUTPUT_DIR / "mdo_mdf_feasible_designs.csv"
    out_pareto = OUTPUT_DIR / "mdo_mdf_pareto_front.csv"
    out_selected = OUTPUT_DIR / "mdo_mdf_selected_representative_designs.csv"

    write_csv(out_all, rows)
    write_csv(out_feasible, feasible_rows)
    write_csv(out_pareto, front)
    write_csv(out_selected, selected)

    print("\nOptimization completed.")
    print(f"Total designs evaluated: {len(rows)}")
    print(f"Feasible designs:        {len(feasible_rows)}")
    print(f"Pareto-front designs:    {len(front)}")
    print(f"Selected designs:        {len(selected)}")

    print("\nSaved files:")
    print(f" - {out_all.resolve()}")
    print(f" - {out_feasible.resolve()}")
    print(f" - {out_pareto.resolve()}")
    print(f" - {out_selected.resolve()}")

    if selected:
        print("\nRepresentative selected designs:")
        for r in selected:
            print("-" * 100)
            print(f"Label: {r['selection_label']}")
            print(
                "z = "
                f"a_CNC={r['a_CNC']}, a_A={r['a_A']}, a_QC={r['a_QC']}, "
                f"r_A={r['r_A']}, r_QC={r['r_QC']}, v_conv={r['v_conv']}, "
                f"u_sched={r['u_sched']}, u_maint={r['u_maint']}, "
                f"u_inspect={r['u_inspect']}, u_energy={r['u_energy']}"
            )
            print(f"Throughput [cartons/h]: {float(r['throughput_cartons_h']):.4f}")
            print(f"Energy [kWh/shift]:     {float(r['E_tot_kWh']):.4f}")
            print(f"Weighted objective:     {float(r['weighted_objective']):.4f}")


if __name__ == "__main__":
    main()
