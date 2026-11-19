"""Baseline design (z0) vs representative Pareto-optimal designs under scenario 7 (all three disruptions), evaluated
with the high-fidelity LHS sample (N_LHS_FINAL, seed LHS_SEED + 1000).

The Max-resilience, Balanced and Max-throughput representatives are read from the saved Pareto front
(results/pareto_front_scenario_7_all_three_disruptions.csv); NSGA-II is not re-run. Outputs are written to separate files.

z0 is the default DesignVector of the deterministic model: {a_CNC: 1, a_A: 1, a_QC: 1, r_A: 2, r_QC: 1, v_conv: 1,
u_sched: 1, u_maint: 1, u_inspect: 2, u_energy: 1}, i.e. the neutral level of every design variable (performance factor
1.0, anchored to REF_THROUGHPUT = 1.625 cartons/h). The MDA-converged nominal throughput of z0 is about 1.70 cartons/h,
because the three-discipline MDA coupling (raw-material feed, queueing between CNC, assembly and QC) acts on top of the
neutral factors.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import csv
import numpy as np
import matplotlib.pyplot as plt

from . import config
from .analysis import evaluate_umda, mean_time_profile
from .design import DV_ORDER
from .exporters import export_workbook, write_csv
from .optimisation import representatives
from .uncertainty import UV_ORDER, distributions_for_design, lhs_unit

SEED = 1042
N = 1500
assert config.N_LHS_FINAL == N and config.LHS_SEED + 1000 == SEED, (
    "This study is specified to use exactly the final-re-evaluation LHS "
    "matrix (N_LHS_FINAL, LHS_SEED+1000) - config.py no longer matches "
    f"N={N}/seed={SEED}; update this assertion deliberately if config.py's "
    "final-stage LHS settings were intentionally changed."
)

Z0: Dict[str, int] = {
    "a_CNC": 1, "a_A": 1, "a_QC": 1, "r_A": 2, "r_QC": 1,
    "v_conv": 1, "u_sched": 1, "u_maint": 1, "u_inspect": 2, "u_energy": 1,
}

_G_KEYS = [
    "g_TH_det", "g_WIP_det", "g_CT_det", "g_E_det", "g_e_carton_det", "g_MDA",
    "g_TH_prob", "g_E_prob", "g_recovery",
]

_PCT_METRICS = [
    "nominal_throughput", "nominal_energy_kWh",
    "disrupted_throughput_mean", "energy_kWh_mean",
    "PL_cartons_mean", "TST_min_mean", "TUT_min_mean",
    "R_phi_tf_mean", "R_phi_end_shift_mean", "P_recover_within_shift",
    "disrupted_throughput_q05", "disrupted_energy_q95",
]

OUT = config.OUTPUT_DIR
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _load_pareto_front_representatives() -> Dict[str, Dict[str, int]]:
    path = OUT / "pareto_front_scenario_7_all_three_disruptions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run umdo_cpps.main.run() first (Scenario 7) "
            "so a saved Pareto front exists to pick representatives from. "
            "This script deliberately does not re-run NSGA-II itself, to "
            "avoid overwriting existing results."
        )
    rows: List[Dict[str, Any]] = list(csv.DictReader(path.open(encoding="utf-8")))
    for r in rows:
        for k in list(r.keys()):
            if k in DV_ORDER:
                r[k] = int(r[k])
            else:
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError):
                    pass
    reps = representatives(rows)
    return {name: {dv: int(reps[name][dv]) for dv in DV_ORDER} for name in reps}


def run():
    pareto_reps = _load_pareto_front_representatives()
    configs = {
        "Baseline ($z_0$)": Z0,
        "Max resilience": pareto_reps["max_resilience"],
        "Balanced": pareto_reps["middle_balanced"],
        "Max throughput": pareto_reps["max_throughput"],
    }

    unit_lhs = lhs_unit(N, len(UV_ORDER), SEED)

    table_rows: List[Dict[str, Any]] = []
    time_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    nominal_th_by_label: Dict[str, float] = {}

    for label, design in configs.items():
        summary, runs, profiles = evaluate_umda(design, unit_lhs, keep_profiles=True)
        dists = distributions_for_design(design)
        nominal_th_by_label[label] = float(summary["nominal_throughput"])

        if not summary["mda_converged"]:
            warnings.append(
                f"{label}: MDA did NOT converge (residual={summary['mda_residual']:.3e})."
            )
        tst_vals = np.asarray([float(r["TST_min"]) for r in runs])
        n_non_recovering = int(np.sum(tst_vals >= config.SHIFT_HOURS * 60.0 - 1e-9))
        if n_non_recovering > 0:
            warnings.append(
                f"{label}: {n_non_recovering}/{N} realizations "
                f"({100*n_non_recovering/N:.1f}%) never achieved sustained "
                f"recovery within the shift (TST_min == censoring sentinel "
                f"{config.SHIFT_HOURS*60:.0f} min)."
            )

        row: Dict[str, Any] = {
            "configuration": label, **design,
            **{k: summary[k] for k in summary if k not in design},
            "n_non_recovering_realizations": n_non_recovering,
            "TTR_CNC_mean_for_design": dists["TTR_CNC"].mean,
            "TTR_Cobot_mean_for_design": dists["TTR_Cobot"].mean,
            "Tcorr_Assembly_mean_for_design": dists["Tcorr_Assembly"].mean,
        }
        for g in _G_KEYS:
            row[f"{g}_satisfied"] = bool(summary[g] <= 0.0)
        table_rows.append(row)
        time_rows.extend({"configuration": label, **r} for r in mean_time_profile(profiles))

    baseline_row = table_rows[0]
    for row in table_rows[1:]:
        for metric in _PCT_METRICS:
            base_v = baseline_row.get(metric)
            if not isinstance(base_v, (int, float)):
                continue
            row[f"{metric}_abs_change_vs_z0"] = row[metric] - base_v
            if base_v != 0:
                row[f"{metric}_pct_change_vs_z0"] = 100.0 * (row[metric] - base_v) / base_v
            # base_v == 0 (e.g. P_recover_within_shift=0.0 at baseline): a
            # relative % change is undefined/infinite - report only the
            # absolute change (set above) instead of a misleading "nan%"/"inf%".

    if not baseline_row["feasible_det"]:
        viol = [g for g in ["g_TH_det", "g_WIP_det", "g_CT_det", "g_E_det", "g_e_carton_det", "g_MDA"] if baseline_row[g] > 0.0]
        warnings.append(
            f"Baseline (z0) violates deterministic/MDA constraint(s): "
            + ", ".join(f"{g}={baseline_row[g]:.4g}" for g in viol)
        )
    if not baseline_row["feasible_uncertain"]:
        viol = [g for g in ["g_TH_prob", "g_E_prob", "g_recovery"] if baseline_row[g] > 0.0]
        warnings.append(
            f"Baseline (z0) violates uncertainty/reliability constraint(s): "
            + ", ".join(f"{g}={baseline_row[g]:.4g}" for g in viol)
        )

    # ---- outputs ----
    write_csv(OUT / "baseline_vs_optimized_scenario_7.csv", table_rows)
    export_workbook(OUT / "baseline_vs_optimized_scenario_7.xlsx", {
        "Summary": table_rows,
        "Time_Profile": time_rows,
    })

    md_path = OUT / "baseline_vs_optimized_scenario_7.md"
    _write_markdown_table(table_rows, warnings, md_path)

    _plot_mean_throughput(time_rows, OUT / "baseline_vs_optimized_mean_throughput.png")
    _plot_cumulative_production(time_rows, nominal_th_by_label, OUT / "baseline_vs_optimized_cumulative_production.png")
    _plot_normalized_resilience_vs_baseline(time_rows, nominal_th_by_label, OUT / "baseline_vs_optimized_normalized_resilience.png")

    print("=" * 80)
    print("Baseline vs. optimized study completed")
    print(f"z0 = {Z0}")
    print(f"LHS: N={N}, seed={SEED} (= config.N_LHS_FINAL / config.LHS_SEED+1000)")
    for row in table_rows:
        print(
            f"{row['configuration']:<18s} TH_nom={row['nominal_throughput']:.3f} "
            f"TH_dis={row['disrupted_throughput_mean']:.3f}+/-{row['disrupted_throughput_std']:.3f} "
            f"E={row['energy_kWh_mean']:.3f}+/-{row['energy_kWh_std']:.3f} "
            f"R_phi(tf)={row['R_phi_tf_mean']:.4f} "
            f"P(recover)={row['P_recover_within_shift']:.3f} "
            f"feasible_overall={row['feasible_overall']}"
        )
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    print(f"\nOutputs: {OUT.resolve()}")
    return table_rows, warnings


def _write_markdown_table(table_rows: List[Dict[str, Any]], warnings: List[str], path: Path):
    cols = [
        "configuration", "nominal_throughput", "nominal_energy_kWh",
        "disrupted_throughput_mean", "disrupted_throughput_std",
        "energy_kWh_mean", "energy_kWh_std",
        "PL_cartons_mean", "TST_min_mean", "TUT_min_mean",
        "R_phi_tf_mean", "R_phi_end_shift_mean",
        "disrupted_throughput_q05", "disrupted_energy_q95", "P_recover_within_shift",
        "feasible_det", "feasible_uncertain", "feasible_overall",
    ]
    lines = ["# Baseline (z0) vs. optimized Pareto solutions - Scenario 7 (ALL_THREE_DISRUPTIONS)", ""]
    lines.append(f"z0 = `{Z0}` (DesignVector() defaults - see module docstring for the 1.625 cartons/h discrepancy note)")
    lines.append(f"LHS: N={N}, seed={SEED} (config.N_LHS_FINAL / config.LHS_SEED+1000)")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for row in table_rows:
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Relative change vs. baseline (z0), %")
    lines.append("")
    lines.append("| configuration | " + " | ".join(_PCT_METRICS) + " |")
    lines.append("|---|" + "|".join(["---"] * len(_PCT_METRICS)) + "|")
    for row in table_rows[1:]:
        cells = []
        for m in _PCT_METRICS:
            pct = row.get(f"{m}_pct_change_vs_z0")
            if pct is not None:
                cells.append(f"{pct:+.1f}%")
            else:
                abs_delta = row.get(f"{m}_abs_change_vs_z0")
                cells.append(f"n/a (z0=0, {abs_delta:+.4g})" if abs_delta is not None else "n/a")
        lines.append(f"| {row['configuration']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- None.")
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_mean_throughput(time_rows, path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    groups: Dict[str, list] = {}
    for r in time_rows:
        groups.setdefault(r["configuration"], []).append(r)
    for i, (label, data) in enumerate(groups.items()):
        data = sorted(data, key=lambda r: r["time_h"])
        ax.plot([r["time_h"] for r in data], [r["throughput_mean"] for r in data], color=COLORS[i % len(COLORS)], label=label)
    ax.set_xlabel("Time [h]"); ax.set_ylabel("Mean throughput [cartons/h]")
    ax.set_title("Mean throughput over time: baseline vs. optimized designs")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def _plot_cumulative_production(time_rows, nominal_th_by_label, path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    groups: Dict[str, list] = {}
    for r in time_rows:
        groups.setdefault(r["configuration"], []).append(r)
    for i, (label, data) in enumerate(groups.items()):
        data = sorted(data, key=lambda r: r["time_h"])
        color = COLORS[i % len(COLORS)]
        ax.plot([r["time_h"] for r in data], [r["cumulative_cartons_mean"] for r in data], color=color, label=label)
        ax.plot([r["time_h"] for r in data], [nominal_th_by_label[label] * r["time_h"] for r in data], color=color, linestyle="--", linewidth=1, alpha=.6)
    ax.plot([], [], color="gray", linestyle="--", linewidth=1, alpha=.6, label="Undisrupted reference (dashed)")
    ax.set_xlabel("Time [h]"); ax.set_ylabel("Cumulative cartons produced")
    ax.set_title("Cumulative production: baseline vs. optimized designs")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def _plot_normalized_resilience_vs_baseline(time_rows, nominal_th_by_label, path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    keep = {"Baseline ($z_0$)", "Max resilience"}
    groups: Dict[str, list] = {}
    for r in time_rows:
        if r["configuration"] in keep:
            groups.setdefault(r["configuration"], []).append(r)
    for label, data in groups.items():
        data = sorted(data, key=lambda r: r["time_h"])
        nominal = nominal_th_by_label[label]
        ax.plot([r["time_h"] for r in data], [r["throughput_mean"] / nominal for r in data], label=label)
    ax.axhline(1.0, linestyle="--", linewidth=1, color="gray")
    ax.set_xlabel("Time [h]"); ax.set_ylabel("Performance (normalized)")
    ax.set_title("Normalized performance: Max resilience vs. Baseline")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


if __name__ == "__main__":
    run()
