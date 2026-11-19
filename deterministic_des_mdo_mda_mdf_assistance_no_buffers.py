
"""
Deterministic DES + MDO/MDA/MDF model for the wooden-box CPPS
with assistance-oriented design variables.

Buffers
-------
The buffer sizes B_WP and B_WB are not design variables: the sensitivity
analysis showed no relevant effect on throughput in the deterministic
configuration. They are fixed model constants:
    FIXED_B_WP = 60
    FIXED_B_WB = 30

Design vector
-------------
z = {
    a_CNC, a_A, a_QC,
    r_A, r_QC,
    v_conv,
    u_sched, u_maint, u_inspect, u_energy
}

Main outputs
------------
- MDA-converged coupling variables:
    q12: CNC -> Assembly flow [box-kits/h]
    q23: Assembly -> QC flow [boxes/h]
- Throughput [cartons/h]
- Total energy [kWh/shift]
- Average WIP approximation
- Cycle time approximation
- Objective values for MDO/MDF:
    f1 = -throughput_cartons_h
    f2 = total_energy_kWh
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


# =============================================================================
# SIMULATION / SYSTEM SETTINGS
# =============================================================================
SHIFT_HOURS = 8.0
SIM_TIME = SHIFT_HOURS * 3600.0

# Deterministic mean times [s]
CNC_SETUP_MEAN = (240.0 + 270.0 + 300.0) / 3.0
CNC_PROCESS_MEAN = (762.0 + 766.0 + 770.0) / 3.0

ASM_SETUP_MEAN = (30.0 + 45.0 + 60.0) / 3.0
ASM_PROCESS_MEAN = (420.0 + 450.0 + 480.0) / 3.0

QC_SETUP_MEAN = (10.0 + 15.0 + 20.0) / 3.0
QC_PROCESS_MEAN = (240.0 + 270.0 + 300.0) / 3.0

RAW_PANELS = 30
BOXES_PER_CARTON = 2
PANELS_PER_BOX = 6
ANGULAR_PER_BOX = 8

# Fixed buffers. Removed from design vector.
FIXED_B_WP = 60
FIXED_B_WB = 30

# Base power values [kW]
P_CNC_BASE = 1.90
P_COBOT_BASE = 0.38
P_QC_BASE = 0.05
P_CONV_BASE = 0.03
P_SCHED_BASE = 0.05
P_MAINT_BASE = 0.02
P_ENERGY_CONTROL_BASE = 0.05

OUTPUT_DIR = Path(".")


# =============================================================================
# DESIGN VECTOR
# =============================================================================
@dataclass(frozen=True)
class DesignVector:
    # Assistance/error-prevention variables
    a_CNC: int = 1
    a_A: int = 1
    a_QC: int = 1

    # Flexibility / automation variables
    r_A: int = 2
    r_QC: int = 1

    # Transport and operational policy variables
    v_conv: int = 1
    u_sched: int = 1
    u_maint: int = 1
    u_inspect: int = 2
    u_energy: int = 1


# Change the candidate design here.
DESIGN_VECTOR = DesignVector(
    a_CNC=2,
    a_A=3,
    a_QC=2,
    r_A=2,
    r_QC=3,
    v_conv=1,
    u_sched=4,
    u_maint=4,
    u_inspect=2,
    u_energy=1,
)


DESIGN_SPACE = {
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


# =============================================================================
# CATEGORY VALUES FROM DES ASSUMPTIONS
# =============================================================================
CATEGORY_VALUES: Dict[str, Dict[int, Tuple[float, float]]] = {
    "a_CNC": {
        1: (1.625 * 1.00, 2.28 + 0.00),
        2: (1.625 * 1.10, 2.28 + 0.05),
        3: (1.625 * 1.20, 2.28 + 0.12),
    },
    "a_A": {
        1: (1.625 * 1.00, 2.28 + 0.00),
        2: (1.625 * 1.15, 2.28 + 0.08),
        3: (1.625 * 1.30, 2.28 + 0.18),
    },
    "a_QC": {
        1: (1.625 * 1.00, 2.28 + 0.00),
        2: (1.625 * 1.12, 2.28 + 0.06),
        3: (1.625 * 1.25, 2.28 + 0.15),
    },
    "r_A": {
        1: (1.381, 2.00),
        2: (1.625, 2.28),
        3: (2.031, 3.10),
    },
    "r_QC": {
        1: (1.625, 2.28),
        2: (1.869, 2.83),
        3: (2.275, 4.48),
    },
    "v_conv": {
        1: (1.625, 2.28),
        2: (2.031, 2.93),
        3: (2.275, 3.28),
        4: (2.519, 4.08),
    },
    "u_sched": {
        1: (1.625, 2.28),
        2: (1.788, 2.33),
        3: (1.950, 2.40),
        4: (2.194, 2.50),
        5: (2.438, 2.65),
    },
    "u_maint": {
        1: (1.625, 2.28),
        2: (1.706, 2.33),
        3: (1.788, 2.40),
        4: (1.950, 2.55),
        5: (2.031, 2.85),
    },
    "u_inspect": {
        1: (1.806, 2.37),
        2: (1.625, 2.28),
        3: (1.451, 2.85),
    },
    "u_energy": {
        1: (1.625, 2.28),
        2: (1.544, 2.35),
        3: (1.495, 2.43),
        4: (1.414, 2.55),
    },
}

ASSISTANCE_POWER_KW: Dict[str, Dict[int, float]] = {
    "a_CNC": {1: 0.00, 2: 0.05, 3: 0.12},
    "a_A": {1: 0.00, 2: 0.08, 3: 0.18},
    "a_QC": {1: 0.00, 2: 0.06, 3: 0.15},
}

REF_THROUGHPUT = 1.625
REF_POWER = 2.28


# =============================================================================
# MODEL SETTINGS FOR MDA/MDF
# =============================================================================
MDA_TOL = 1e-6
MDA_MAX_ITER = 100
MDA_RELAXATION = 0.50

# Constraint thresholds.
THROUGHPUT_MIN = 1.0       # cartons/h
WIP_MAX = 25.0             # units
CT_MAX = 3600.0            # s/carton
E_TOT_MAX = 30.0           # kWh/shift
E_CARTON_MAX = 3.0         # kWh/carton

# Reference values for single weighted objective
TH_REF = 1.625
E_REF = 20.0
W_TH = 0.50
W_E = 0.50


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def category_factor(var_name: str, category: int) -> Tuple[float, float]:
    th, p = CATEGORY_VALUES[var_name][int(category)]
    performance_factor = th / REF_THROUGHPUT
    power_factor = p / REF_POWER
    return performance_factor, power_factor


def time_multiplier_from_performance(performance_factor: float) -> float:
    return 1.0 / max(0.20, performance_factor)


def clamp_int(value: int, allowed: List[int], name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be in {allowed}, got {value}")


def validate_design(z: DesignVector) -> None:
    for name, values in DESIGN_SPACE.items():
        clamp_int(getattr(z, name), values, name)


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    if abs(den) < 1e-12:
        return default
    return num / den


# =============================================================================
# MULTIPLIER MODEL
# =============================================================================
@dataclass
class Multipliers:
    cnc_time_mult: float
    asm_time_mult: float
    qc_time_mult: float
    transfer_delay_s: float

    cnc_power_mult: float
    asm_power_mult: float
    qc_power_mult: float
    conv_power_mult: float
    sched_power_mult: float
    maint_power_mult: float
    energy_control_power_mult: float

    p_assist_cnc_kw: float
    p_assist_a_kw: float
    p_assist_qc_kw: float

    system_performance_factor: float


def compute_multipliers(z: DesignVector) -> Multipliers:
    validate_design(z)

    a_cnc_perf, a_cnc_pow = category_factor("a_CNC", z.a_CNC)
    a_a_perf, a_a_pow = category_factor("a_A", z.a_A)
    a_qc_perf, a_qc_pow = category_factor("a_QC", z.a_QC)

    r_a_perf, r_a_pow = category_factor("r_A", z.r_A)
    r_qc_perf, r_qc_pow = category_factor("r_QC", z.r_QC)
    v_perf, v_pow = category_factor("v_conv", z.v_conv)
    sched_perf, sched_pow = category_factor("u_sched", z.u_sched)
    maint_perf, maint_pow = category_factor("u_maint", z.u_maint)
    inspect_perf, inspect_pow = category_factor("u_inspect", z.u_inspect)
    energy_perf, energy_pow = category_factor("u_energy", z.u_energy)

    cnc_time_mult = (
        time_multiplier_from_performance(a_cnc_perf)
        * time_multiplier_from_performance(maint_perf) ** 0.50
        * time_multiplier_from_performance(sched_perf) ** 0.25
        * time_multiplier_from_performance(energy_perf) ** 0.25
    )

    asm_time_mult = (
        time_multiplier_from_performance(a_a_perf)
        * time_multiplier_from_performance(r_a_perf)
        * time_multiplier_from_performance(maint_perf) ** 0.50
        * time_multiplier_from_performance(sched_perf) ** 0.25
        * time_multiplier_from_performance(energy_perf) ** 0.25
    )

    qc_time_mult = (
        time_multiplier_from_performance(a_qc_perf)
        * time_multiplier_from_performance(r_qc_perf)
        * time_multiplier_from_performance(inspect_perf)
        * time_multiplier_from_performance(sched_perf) ** 0.20
        * time_multiplier_from_performance(energy_perf) ** 0.20
    )

    transfer_delay_s = {1: 20.0, 2: 8.0, 3: 4.0, 4: 2.0}[z.v_conv]

    # System-level performance factor used for MDO-oriented adjusted throughput.
    # Weights sum to 1.0.
    system_performance_factor = (
        a_cnc_perf ** 0.12
        * a_a_perf ** 0.12
        * a_qc_perf ** 0.10
        * r_a_perf ** 0.13
        * r_qc_perf ** 0.12
        * v_perf ** 0.10
        * sched_perf ** 0.12
        * maint_perf ** 0.09
        * inspect_perf ** 0.05
        * energy_perf ** 0.05
    )

    return Multipliers(
        cnc_time_mult=cnc_time_mult,
        asm_time_mult=asm_time_mult,
        qc_time_mult=qc_time_mult,
        transfer_delay_s=transfer_delay_s,
        cnc_power_mult=a_cnc_pow * maint_pow * sched_pow * energy_pow,
        asm_power_mult=a_a_pow * r_a_pow * maint_pow * energy_pow,
        qc_power_mult=a_qc_pow * r_qc_pow * inspect_pow * energy_pow,
        conv_power_mult=v_pow * energy_pow,
        sched_power_mult=sched_pow,
        maint_power_mult=maint_pow,
        energy_control_power_mult=energy_pow,
        p_assist_cnc_kw=ASSISTANCE_POWER_KW["a_CNC"][z.a_CNC],
        p_assist_a_kw=ASSISTANCE_POWER_KW["a_A"][z.a_A],
        p_assist_qc_kw=ASSISTANCE_POWER_KW["a_QC"][z.a_QC],
        system_performance_factor=system_performance_factor,
    )


# =============================================================================
# COUPLINGS AND DISCIPLINE OUTPUTS
# =============================================================================
@dataclass
class Couplings:
    q12: float = 1.0  # CNC -> Assembly flow [box-kits/h]
    q23: float = 1.0  # Assembly -> QC flow [boxes/h]


@dataclass
class DisciplineOutput:
    data: Dict[str, float]


# =============================================================================
# DISCIPLINE CLASSES
# =============================================================================
class CNCDiscipline:
    """
    Discipline 1: CNC machining.

    Local design variables:
        a_CNC, u_maint

    Shared/indirect variables:
        v_conv, u_sched, u_energy

    Output coupling:
        q12: output flow of box-kits from CNC to Assembly [box-kits/h]
    """

    def evaluate(self, z: DesignVector, y: Couplings, m: Multipliers) -> Tuple[float, DisciplineOutput]:
        cycle_s = (CNC_SETUP_MEAN + CNC_PROCESS_MEAN) * m.cnc_time_mult + m.transfer_delay_s
        cap_box_kits_h = 3600.0 / cycle_s

        # Raw material limit: each raw panel yields one box-kit.
        raw_limit_box_kits_h = RAW_PANELS / SHIFT_HOURS
        q12_new = min(cap_box_kits_h, raw_limit_box_kits_h)

        busy_time_h = min(q12_new * SHIFT_HOURS, RAW_PANELS) * cycle_s / 3600.0
        e_cnc = P_CNC_BASE * m.cnc_power_mult * busy_time_h

        out = DisciplineOutput({
            "N_mach_box_kits": q12_new * SHIFT_HOURS,
            "TH_CNC_box_kits_h": q12_new,
            "t_proc_CNC_s": cycle_s,
            "U_CNC": min(1.0, busy_time_h / SHIFT_HOURS),
            "E_CNC_kWh": e_cnc,
        })

        return q12_new, out


class AssemblyDiscipline:
    """
    Discipline 2: collaborative assembly.

    Coupling input:
        q12 from CNC

    Output coupling:
        q23: assembled boxes sent to QC [boxes/h]
    """

    def evaluate(self, z: DesignVector, y: Couplings, m: Multipliers) -> Tuple[float, DisciplineOutput]:
        cycle_s = (ASM_SETUP_MEAN + ASM_PROCESS_MEAN) * m.asm_time_mult + m.transfer_delay_s
        cap_boxes_h = 3600.0 / cycle_s

        q23_new = min(y.q12, cap_boxes_h)

        busy_time_h = q23_new * SHIFT_HOURS * cycle_s / 3600.0
        e_a = P_COBOT_BASE * m.asm_power_mult * busy_time_h

        out = DisciplineOutput({
            "N_box": q23_new * SHIFT_HOURS,
            "TH_A_boxes_h": q23_new,
            "t_proc_A_s": cycle_s,
            "U_A": min(1.0, busy_time_h / SHIFT_HOURS),
            "E_A_kWh": e_a,
        })

        return q23_new, out


class QCDiscipline:
    """
    Discipline 3: QC and packaging.

    Coupling input:
        q23 from Assembly

    Final output:
        TH_QC_cartons_h
    """

    def evaluate(self, z: DesignVector, y: Couplings, m: Multipliers) -> DisciplineOutput:
        cycle_s_per_carton = ((QC_SETUP_MEAN + QC_PROCESS_MEAN) * m.qc_time_mult) * BOXES_PER_CARTON
        cap_cartons_h = 3600.0 / cycle_s_per_carton

        incoming_cartons_h = y.q23 / BOXES_PER_CARTON
        th_cartons_h_des = min(incoming_cartons_h, cap_cartons_h)

        # MDO-oriented adjusted throughput, consistent with sensitivity-analysis code.
        th_cartons_h_adjusted = th_cartons_h_des * m.system_performance_factor

        n_cartons = th_cartons_h_des * SHIFT_HOURS
        busy_time_h = n_cartons * cycle_s_per_carton / 3600.0

        e_qc = P_QC_BASE * m.qc_power_mult * busy_time_h

        out = DisciplineOutput({
            "N_good_boxes": n_cartons * BOXES_PER_CARTON,
            "N_carton": n_cartons,
            "TH_QC_cartons_h_DES": th_cartons_h_des,
            "TH_QC_cartons_h_adjusted": th_cartons_h_adjusted,
            "t_proc_QC_s_per_carton": cycle_s_per_carton,
            "U_QC": min(1.0, busy_time_h / SHIFT_HOURS),
            "E_QC_kWh": e_qc,
        })

        return out


# =============================================================================
# MDA LOOP WITH FIXED-POINT ITERATION
# =============================================================================
@dataclass
class MDAResult:
    converged: bool
    iterations: int
    final_couplings: Couplings
    residual_history: List[float]
    outputs: Dict[str, float]


class MDAProblem:
    def __init__(self, z: DesignVector):
        self.z = z
        self.m = compute_multipliers(z)
        self.d1 = CNCDiscipline()
        self.d2 = AssemblyDiscipline()
        self.d3 = QCDiscipline()

    def solve(self) -> MDAResult:
        y = Couplings(q12=1.0, q23=1.0)
        residual_history: List[float] = []
        converged = False
        outputs: Dict[str, float] = {}

        for k in range(1, MDA_MAX_ITER + 1):
            old_q12, old_q23 = y.q12, y.q23

            # Gauss-Seidel sequence:
            # 1) CNC updates q12
            q12_calc, out1 = self.d1.evaluate(self.z, y, self.m)
            y.q12 = (1.0 - MDA_RELAXATION) * y.q12 + MDA_RELAXATION * q12_calc

            # 2) Assembly uses updated q12 and updates q23
            q23_calc, out2 = self.d2.evaluate(self.z, y, self.m)
            y.q23 = (1.0 - MDA_RELAXATION) * y.q23 + MDA_RELAXATION * q23_calc

            # 3) QC uses updated q23
            out3 = self.d3.evaluate(self.z, y, self.m)

            residual = math.sqrt((y.q12 - old_q12) ** 2 + (y.q23 - old_q23) ** 2)
            residual_history.append(residual)

            outputs = {}
            outputs.update(out1.data)
            outputs.update(out2.data)
            outputs.update(out3.data)

            if residual < MDA_TOL:
                converged = True
                break

        # Add energy and system-level KPIs from converged response.
        outputs.update(self._compute_system_kpis(y, outputs, converged))

        return MDAResult(
            converged=converged,
            iterations=len(residual_history),
            final_couplings=y,
            residual_history=residual_history,
            outputs=outputs,
        )

    def _compute_system_kpis(self, y: Couplings, outputs: Dict[str, float], converged: bool) -> Dict[str, float]:
        # Energy components
        e_cnc = outputs["E_CNC_kWh"]
        e_a = outputs["E_A_kWh"]
        e_qc = outputs["E_QC_kWh"]

        # Approximate transport active time based on units transferred.
        transferred_units = (y.q12 + y.q23) * SHIFT_HOURS
        transport_active_h = transferred_units * self.m.transfer_delay_s / 3600.0
        e_transport = P_CONV_BASE * self.m.conv_power_mult * transport_active_h

        e_sched = P_SCHED_BASE * self.m.sched_power_mult * SHIFT_HOURS
        e_maint = P_MAINT_BASE * self.m.maint_power_mult * SHIFT_HOURS
        e_energy_control = P_ENERGY_CONTROL_BASE * self.m.energy_control_power_mult * SHIFT_HOURS

        e_assist_cnc = self.m.p_assist_cnc_kw * SHIFT_HOURS
        e_assist_a = self.m.p_assist_a_kw * SHIFT_HOURS
        e_assist_qc = self.m.p_assist_qc_kw * SHIFT_HOURS
        e_assistance = e_assist_cnc + e_assist_a + e_assist_qc

        e_tot = (
            e_cnc + e_a + e_qc + e_transport
            + e_sched + e_maint + e_energy_control
            + e_assistance
        )

        th_des = outputs["TH_QC_cartons_h_DES"]
        th_adjusted = outputs["TH_QC_cartons_h_adjusted"]
        n_cartons = outputs["N_carton"]

        # Deterministic approximations for WIP and CT.
        # WIP is estimated from flow imbalance between stages.
        wip_panels = max(0.0, y.q12 - y.q23) * SHIFT_HOURS
        wip_boxes = max(0.0, y.q23 - (th_des * BOXES_PER_CARTON)) * SHIFT_HOURS
        avg_wip = min(FIXED_B_WP + FIXED_B_WB, wip_panels + wip_boxes)

        ct_s_per_carton = 3600.0 / th_adjusted if th_adjusted > 1e-12 else float("inf")
        e_carton = safe_div(e_tot, max(1e-9, n_cartons), default=float("inf"))

        # Objective functions
        f1 = -th_adjusted
        f2 = e_tot
        weighted_objective = W_TH * (-th_adjusted / TH_REF) + W_E * (e_tot / E_REF)

        # Constraint violations. 0 means feasible.
        violations = {
            "viol_TH_min": max(0.0, THROUGHPUT_MIN - th_adjusted),
            "viol_WIP_max": max(0.0, avg_wip - WIP_MAX),
            "viol_CT_max": max(0.0, ct_s_per_carton - CT_MAX),
            "viol_E_tot_max": max(0.0, e_tot - E_TOT_MAX),
            "viol_e_carton_max": max(0.0, e_carton - E_CARTON_MAX),
            "viol_MDA_convergence": 0.0 if converged else 1.0,
        }

        return {
            "throughput_cartons_h": th_adjusted,
            "throughput_cartons_h_DES": th_des,
            "throughput_cartons_h_adjusted": th_adjusted,
            "system_performance_factor": self.m.system_performance_factor,
            "N_carton_shift": n_cartons,
            "avg_WIP_units": avg_wip,
            "cycle_time_s_per_carton": ct_s_per_carton,
            "E_CNC_kWh": e_cnc,
            "E_Assembly_kWh": e_a,
            "E_QC_kWh": e_qc,
            "E_Transport_kWh": e_transport,
            "E_Scheduling_kWh": e_sched,
            "E_Maintenance_kWh": e_maint,
            "E_EnergyControl_kWh": e_energy_control,
            "E_Assistance_CNC_kWh": e_assist_cnc,
            "E_Assistance_Assembly_kWh": e_assist_a,
            "E_Assistance_QC_kWh": e_assist_qc,
            "E_Assistance_Total_kWh": e_assistance,
            "E_tot_kWh": e_tot,
            "average_power_kW": e_tot / SHIFT_HOURS,
            "e_carton_kWh_per_carton": e_carton,
            "f1_performance_min_form": f1,
            "f2_sustainability_energy": f2,
            "weighted_objective": weighted_objective,
            **violations,
        }


# =============================================================================
# MDF EVALUATION WRAPPER
# =============================================================================
def evaluate_mdf(z: DesignVector) -> Dict[str, Any]:
    validate_design(z)
    problem = MDAProblem(z)
    mda = problem.solve()

    row: Dict[str, Any] = {
        **asdict(z),
        "MDA_converged": mda.converged,
        "MDA_iterations": mda.iterations,
        "q12_box_kits_h": mda.final_couplings.q12,
        "q23_boxes_h": mda.final_couplings.q23,
        "MDA_final_residual": mda.residual_history[-1] if mda.residual_history else float("nan"),
        "MDA_residual_history": str(mda.residual_history),
    }
    row.update(mda.outputs)
    return row


# =============================================================================
# BASIC DESIGN SCREENING / EXPORT HELPERS
# =============================================================================
def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def one_at_a_time_mdf_sweep(base: DesignVector) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for var, values in DESIGN_SPACE.items():
        for value in values:
            z_dict = asdict(base)
            z_dict[var] = value
            z = DesignVector(**z_dict)
            row = evaluate_mdf(z)
            row["experiment"] = "MDF_one_at_a_time"
            row["varied_variable"] = var
            row["varied_value"] = value
            rows.append(row)
    return rows


def print_report(row: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("DETERMINISTIC DES + MDO/MDA/MDF RESULTS")
    print("=" * 100)

    print("\nDesign vector z (B_WP and B_WB removed):")
    for key in DESIGN_SPACE.keys():
        print(f"{key:>12s}: {row[key]}")

    print("-" * 100)
    print(f"MDA converged:                 {row['MDA_converged']}")
    print(f"MDA iterations:                {row['MDA_iterations']}")
    print(f"Final q12 [box-kits/h]:        {row['q12_box_kits_h']:.6f}")
    print(f"Final q23 [boxes/h]:           {row['q23_boxes_h']:.6f}")
    print("-" * 100)
    print("System KPIs:")
    print(f"Throughput adjusted [cartons/h]: {row['throughput_cartons_h_adjusted']:.6f}")
    print(f"Throughput DES [cartons/h]:      {row['throughput_cartons_h_DES']:.6f}")
    print(f"N_carton [cartons/shift]:        {row['N_carton_shift']:.6f}")
    print(f"Average WIP [units]:             {row['avg_WIP_units']:.6f}")
    print(f"Cycle time [s/carton]:           {row['cycle_time_s_per_carton']:.6f}")
    print(f"E_tot [kWh]:                     {row['E_tot_kWh']:.6f}")
    print(f"e_carton [kWh/carton]:           {row['e_carton_kWh_per_carton']:.6f}")
    print("-" * 100)
    print("Objectives:")
    print(f"f1 = -TH:                         {row['f1_performance_min_form']:.6f}")
    print(f"f2 = E_tot:                       {row['f2_sustainability_energy']:.6f}")
    print(f"Weighted objective:               {row['weighted_objective']:.6f}")
    print("-" * 100)
    print("Constraint violations:")
    for key in [
        "viol_TH_min", "viol_WIP_max", "viol_CT_max",
        "viol_E_tot_max", "viol_e_carton_max", "viol_MDA_convergence",
    ]:
        print(f"{key:>24s}: {row[key]:.6f}")


def main() -> None:
    # 1) Evaluate selected design
    selected = evaluate_mdf(DESIGN_VECTOR)
    print_report(selected)
    out_selected = OUTPUT_DIR / "deterministic_des_mdo_mda_mdf_selected_no_buffers.csv"
    write_csv(out_selected, [selected])
    print(f"\nSaved selected MDF result to: {out_selected.resolve()}")

    # 2) Optional one-at-a-time MDF sweep
    rows = one_at_a_time_mdf_sweep(DESIGN_VECTOR)
    out_sweep = OUTPUT_DIR / "deterministic_des_mdo_mda_mdf_oat_no_buffers.csv"
    write_csv(out_sweep, rows)
    print(f"Saved MDF one-at-a-time sweep to: {out_sweep.resolve()}")


if __name__ == "__main__":
    main()
