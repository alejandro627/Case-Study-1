from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from . import config
from .analysis import evaluate_umda
from .disruption_sensitivity import REFERENCE_DESIGN
from .exporters import write_csv
from .scenarios import get_scenario
from .uncertainty import UV_ORDER, lhs_unit

# LHS sample sizes and number of independent replications (LHS seeds) per size. The spread across the replications
# estimates the sampling error at each N (about 1/sqrt(N)).
SAMPLE_SIZES = [200, 500, 1000, 1500, 3000, 5000]
N_REPLICATIONS = 8

# Continuous objective and performance metrics (mean +/- SEM / CI).
METRICS = [
    "disrupted_throughput_mean", "energy_kWh_mean", "PL_cartons_mean",
    "TST_min_mean", "TUT_min_mean", "R_phi_tf_mean",
]
# Stochastic constraint margins (built from percentiles and probabilities over the LHS realisations, see
# analysis.evaluate_umda()). The deterministic constraints do not depend on the LHS sample and are excluded.
CONSTRAINT_MARGINS = ["g_TH_prob", "g_E_prob", "g_recovery"]
METRIC_LABELS = {
    "disrupted_throughput_mean": "Disrupted throughput [cartons/h]",
    "energy_kWh_mean": "Energy [kWh]",
    "PL_cartons_mean": "PL [cartons]",
    "TST_min_mean": "TST [min]",
    "TUT_min_mean": "TUT [min]",
    "R_phi_tf_mean": r"$R_\phi(t_f)$ [-]",
}
# Relative SEM target: one diagnostic among several (relative error is unreliable near zero).
RELATIVE_SEM_TARGET = 0.02


def run(
    design: Optional[Mapping[str, int]] = None,
    sample_sizes: Sequence[int] = SAMPLE_SIZES,
    n_replications: int = N_REPLICATIONS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """For each sample size N, evaluate the same design with n_replications independent LHS seeds and record the
        continuous METRICS, the three stochastic CONSTRAINT_MARGINS and the feasible_overall classification.
        Returns (raw_rows, summary_rows).

        Per continuous metric, summary_rows gives the mean over the replications ('_estimate'), standard deviation, SEM,
        relative SEM, a 95 % Student-t confidence interval and the absolute difference to the estimate at the largest N
        (reference). For feasibility it gives feasible_overall_classification_stability: the share of replications
        whose feasibility decision agrees with the majority (1.0 = all agree).
    """
    design = dict(design) if design is not None else dict(REFERENCE_DESIGN)
    raw_rows: List[Dict[str, Any]] = []

    for n in sample_sizes:
        for rep in range(n_replications):
            seed = config.LHS_SEED + n * 1000 + rep
            lhs = lhs_unit(n, len(UV_ORDER), seed)
            summary, _, _ = evaluate_umda(design, lhs, keep_profiles=False)
            row = {"N": n, "replication": rep, "seed": seed}
            row.update({m: float(summary[m]) for m in METRICS})
            row.update({c: float(summary[c]) for c in CONSTRAINT_MARGINS})
            row["feasible_overall"] = bool(summary["feasible_overall"])
            row["feasible_det"] = bool(summary["feasible_det"])
            row["feasible_uncertain"] = bool(summary["feasible_uncertain"])
            raw_rows.append(row)

    largest_n = max(sample_sizes)
    reference_rows = [r for r in raw_rows if r["N"] == largest_n]
    reference_estimate = {m: float(np.mean([r[m] for r in reference_rows])) for m in METRICS + CONSTRAINT_MARGINS}

    summary_rows: List[Dict[str, Any]] = []
    for n in sample_sizes:
        rows_n = [r for r in raw_rows if r["N"] == n]
        row: Dict[str, Any] = {"N": n, "n_replications": n_replications}
        for m in METRICS + CONSTRAINT_MARGINS:
            vals = np.asarray([r[m] for r in rows_n])
            mean_of_means = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            sem = sd / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            # 95 % confidence interval of the mean (Student-t, df = n_replications - 1).
            if len(vals) > 1:
                tcrit = float(stats.t.ppf(0.975, df=len(vals) - 1))
            else:
                tcrit = float("nan")
            ci_half_width = tcrit * sem
            row[f"{m}_estimate"] = mean_of_means
            row[f"{m}_std_across_replications"] = sd
            row[f"{m}_sem_across_replications"] = sem
            row[f"{m}_relative_sem"] = sem / abs(mean_of_means) if abs(mean_of_means) > 1e-9 else float("nan")
            row[f"{m}_ci95_half_width"] = ci_half_width
            row[f"{m}_ci95_low"] = mean_of_means - ci_half_width
            row[f"{m}_ci95_high"] = mean_of_means + ci_half_width
            row[f"{m}_abs_diff_from_largest_N"] = abs(mean_of_means - reference_estimate[m])

        # Stability of the feasibility classification (see run()).
        for flag in ("feasible_overall", "feasible_det", "feasible_uncertain"):
            votes = [r[flag] for r in rows_n]
            majority = sum(votes) >= len(votes) / 2
            stability = sum(1 for v in votes if v == majority) / len(votes)
            row[f"{flag}_majority"] = bool(majority)
            row[f"{flag}_classification_stability"] = float(stability)

        summary_rows.append(row)

    return raw_rows, summary_rows


def recommend_n(
    summary_rows: Sequence[Mapping[str, Any]],
    metric: str,
    relative_sem_target: float = RELATIVE_SEM_TARGET,
    abs_diff_target: Optional[float] = None,
) -> Optional[int]:
    """Smallest tested N that meets the relative-SEM target and, if given, the absolute-difference target with
        respect to the largest N. abs_diff_target=None uses only the relative criterion.
    """
    for row in sorted(summary_rows, key=lambda r: r["N"]):
        rel_ok = row[f"{metric}_relative_sem"] <= relative_sem_target
        abs_ok = True if abs_diff_target is None else row[f"{metric}_abs_diff_from_largest_N"] <= abs_diff_target
        if rel_ok and abs_ok:
            return int(row["N"])
    return None


def recommend_n_for_feasibility(summary_rows: Sequence[Mapping[str, Any]], flag: str = "feasible_overall") -> Optional[int]:
    """Smallest tested N from which the feasibility classification is unanimous for this and every larger tested N."""
    rows_sorted = sorted(summary_rows, key=lambda r: r["N"])
    for i, row in enumerate(rows_sorted):
        if all(r[f"{flag}_classification_stability"] >= 1.0 for r in rows_sorted[i:]):
            return int(row["N"])
    return None


def plot(summary_rows: Sequence[Mapping[str, Any]], path, title: str):
    rows_sorted = sorted(summary_rows, key=lambda r: r["N"])
    ns = [row["N"] for row in rows_sorted]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, metric in zip(axes.flat[:6], METRICS):
        means = [row[f"{metric}_estimate"] for row in rows_sorted]
        sems = [row[f"{metric}_sem_across_replications"] for row in rows_sorted]
        ax.errorbar(ns, means, yerr=sems, marker="o", capsize=4)
        ax.axhline(means[-1], linestyle="--", linewidth=1, color="gray")
        ax.set_xscale("log")
        ax.set_xlabel("LHS sample size N")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(METRIC_LABELS[metric])
    # Feasibility-classification stability vs N and the absolute difference of the stochastic constraint margins to
    # the reference.
    ax = axes.flat[6]
    for flag, marker in (("feasible_overall", "o"), ("feasible_uncertain", "s")):
        stabilities = [row[f"{flag}_classification_stability"] for row in rows_sorted]
        ax.plot(ns, stabilities, marker=marker, label=flag)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("LHS sample size N")
    ax.set_ylabel("Classification stability (1.0 = unanimous)")
    ax.set_title("Feasibility classification stability")
    ax.legend(fontsize=7)

    ax = axes.flat[7]
    for c in CONSTRAINT_MARGINS:
        diffs = [row[f"{c}_abs_diff_from_largest_N"] for row in rows_sorted]
        ax.plot(ns, diffs, marker="o", label=c)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("LHS sample size N")
    ax.set_ylabel("Abs. diff from largest-N estimate")
    ax.set_title("Stochastic constraint margins: absolute convergence")
    ax.legend(fontsize=7)

    axes.flat[8].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def export(raw_rows: Sequence[Mapping[str, Any]], summary_rows: Sequence[Mapping[str, Any]], suffix: str, out_dir=None) -> None:
    out = out_dir or config.OUTPUT_DIR
    write_csv(out / f"lhs_convergence_raw_{suffix}.csv", raw_rows)
    write_csv(out / f"lhs_convergence_summary_{suffix}.csv", summary_rows)


def suffix_for_current_scenario() -> str:
    scenario = get_scenario(config.SCENARIO_MODE)
    return f"scenario_{config.SCENARIO_MODE}_{scenario['name'].lower()}"
