from __future__ import annotations
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import matplotlib.pyplot as plt
from .optimisation import display_label

def hypervolume_convergence(rows: Sequence[Mapping], path: Path, title: str):
    rows = sorted(rows, key=lambda r: r["n_gen"])
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot([r["n_gen"] for r in rows], [r["hypervolume"] for r in rows], color="tab:blue", label="Hypervolume")
    ax1.set_xlabel("Generation"); ax1.set_ylabel("Hypervolume", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot([r["n_gen"] for r in rows], [r["n_nds"] for r in rows], color="tab:gray", linestyle="--", alpha=.6, label="n_nds")
    ax2.set_ylabel("Non-dominated count (n_nds)", color="tab:gray")
    ax2.tick_params(axis="y", labelcolor="tab:gray")
    ax1.set_title(title); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)

def convergence_diagnostics(rows: Sequence[Mapping], path: Path, title: str):
    """Four-panel convergence figure from the per-generation diagnostics (nsga2_generations.csv): hypervolume,
        feasible / non-dominated feasible counts, best and mean objective values (minimisation convention) and
        population diversity vs generation.
    """
    rows = sorted(rows, key=lambda r: r["n_gen"])
    gens = [r["n_gen"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(gens, [r["hypervolume"] for r in rows], marker="o", color="tab:blue")
    ax.set_xlabel("Generation"); ax.set_ylabel("Hypervolume (fixed ref. point)")
    ax.set_title("Hypervolume vs. generation")

    ax = axes[0, 1]
    ax.plot(gens, [r["n_feasible_in_population"] for r in rows], marker="o", label="Feasible in population")
    ax.plot(gens, [r["n_nondominated_feasible"] for r in rows], marker="s", label="Non-dominated & feasible")
    ax.plot(gens, [r["pop_size"] for r in rows], linestyle=":", color="gray", label="Population size")
    ax.set_xlabel("Generation"); ax.set_ylabel("Count")
    ax.set_title("Feasible / non-dominated solutions vs. generation")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    obj_names = [("F0_neg_throughput", "-TH"), ("F1_energy", "E"), ("F2_neg_R_phi_tf", "-R_phi(tf)")]
    for key, label in obj_names:
        ax.plot(gens, [r[f"{key}_best"] for r in rows], marker="o", label=f"{label} best")
        ax.plot(gens, [r[f"{key}_mean"] for r in rows], linestyle="--", label=f"{label} mean")
    ax.set_xlabel("Generation"); ax.set_ylabel("Objective value (minimize convention)")
    ax.set_title("Objective evolution vs. generation")
    ax.legend(fontsize=6, ncol=2)

    ax = axes[1, 1]
    ax.plot(gens, [r["n_unique_designs_in_population"] for r in rows], marker="o", color="tab:purple")
    ax.plot(gens, [r["pop_size"] for r in rows], linestyle=":", color="gray", label="Population size")
    ax.set_xlabel("Generation"); ax.set_ylabel("Unique integer designs in population")
    ax.set_title("Discrete population diversity vs. generation")
    ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def pareto_3d(rows: Sequence[Mapping], reps: Mapping[str,Mapping], path: Path, title: str,
              resilience_key: str = "R_phi_tf_mean", resilience_label: str = r"Mean $R_\phi(t_f)$ [-]",
              lower_is_better: bool = False):
    # The resilience axis defaults to R_phi(t_f) (maximised); resilience_key, label and lower_is_better allow other
    # metrics (e.g., TTR_mean, lower is better).
    fig=plt.figure(figsize=(10,8))
    ax=fig.add_subplot(111,projection="3d")
    x=[r["disrupted_throughput_mean"] for r in rows]
    y=[r["energy_kWh_mean"] for r in rows]
    z=[r[resilience_key] for r in rows]
    ax.scatter(x,y,z,alpha=.65)
    markers={"max_throughput":"^","min_energy":"s","max_resilience":"*","middle_balanced":"D"}
    for name,r in reps.items():
        ax.scatter([r["disrupted_throughput_mean"]],[r["energy_kWh_mean"]],[r[resilience_key]],
                   marker=markers[name],s=130,label=display_label(name))
    ax.set_xlabel("Mean disrupted throughput [cartons/h]")
    ax.set_ylabel("Mean energy [kWh/shift]")
    label = resilience_label + (" - smaller is better" if lower_is_better else "")
    ax.set_zlabel(label)
    ax.set_title(title); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig)

def resilience_time(rows: Sequence[Mapping], path: Path, title: str):
    fig,ax=plt.subplots(figsize=(10,6))
    groups={}
    for r in rows:
        groups.setdefault(r["representative_solution"],[]).append(r)
    for name,data in groups.items():
        data=sorted(data,key=lambda r:r["time_h"])
        ax.plot([r["time_h"] for r in data],[r["throughput_mean"] for r in data],label=display_label(name))
    ax.set_xlabel("Time [h]"); ax.set_ylabel("Mean throughput [cartons/h]")
    ax.set_title(title); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig)

def resilience_time_normalized(rows: Sequence[Mapping], path: Path, title: str):
    fig,ax=plt.subplots(figsize=(10,6))
    groups={}
    for r in rows:
        groups.setdefault(r["representative_solution"],[]).append(r)
    for name,data in groups.items():
        data=sorted(data,key=lambda r:r["time_h"])
        ax.plot([r["time_h"] for r in data],[r["throughput_normalized"] for r in data],label=display_label(name))
    ax.axhline(1.0,linestyle="--",linewidth=1,color="gray")
    ax.set_xlabel("Time [h]"); ax.set_ylabel("Performance")
    ax.set_title(title); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig)

def cumulative_production_time(rows: Sequence[Mapping], path: Path, title: str):
    # Cumulative cartons (solid) and the undisrupted reference at the design's nominal rate (dashed, same colour); the
    # gap is the production lost to the disruptions (no catch-up is modelled).
    fig,ax=plt.subplots(figsize=(10,6))
    groups={}
    for r in rows:
        groups.setdefault(r["representative_solution"],[]).append(r)
    colors=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i,(name,data) in enumerate(groups.items()):
        data=sorted(data,key=lambda r:r["time_h"])
        color=colors[i%len(colors)]
        ax.plot([r["time_h"] for r in data],[r["cumulative_cartons_mean"] for r in data],
                color=color,label=display_label(name))
        ax.plot([r["time_h"] for r in data],[r["cumulative_cartons_reference"] for r in data],
                color=color,linestyle="--",linewidth=1,alpha=0.6)
    ax.plot([],[],color="gray",linestyle="--",linewidth=1,alpha=0.6,label="Undisrupted reference (dashed)")
    ax.set_xlabel("Time [h]"); ax.set_ylabel("Cumulative cartons produced")
    ax.set_title(title); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig)

def resilience_trajectory_annotated(mean_profile_rows: Sequence[Mapping], events: Sequence[Mapping], nominal_th: float, path: Path, title: str):
    """Mean disrupted throughput trajectory with the nominal level, the recovery threshold, disruption onsets and
        ends, t_d (maximum degradation), t_r (sustained recovery) and TTR, TST, PL and recovery rate annotated. The
        metrics are recomputed on the plotted mean trajectory, so the annotations match the curve.
    """
    from . import config
    from .metrics import calculate
    profile = [{"time_min": r["time_min"], "throughput_cartons_h": r["throughput_mean"]} for r in mean_profile_rows]
    m = calculate(profile, events, nominal_th)
    times_h = [r["time_h"] for r in mean_profile_rows]
    values = [r["throughput_mean"] for r in mean_profile_rows]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(times_h, values, color="tab:blue", label="Mean disrupted throughput")
    ax.axhline(nominal_th, color="gray", linestyle=":", linewidth=1, label="Nominal throughput")
    ax.axhline(config.RECOVERY_THRESHOLD_FRACTION * nominal_th, color="tab:green", linestyle="--", linewidth=1,
               label=f"{config.RECOVERY_THRESHOLD_FRACTION*100:.0f}% recovery threshold")

    sorted_events = sorted(events, key=lambda e: float(e["start_min"]))
    for e in sorted_events:
        ax.axvline(e["start_min"] / 60.0, color="tab:red", linestyle="-", alpha=.4)
        ax.axvline(e["end_min"] / 60.0, color="tab:orange", linestyle="-", alpha=.4)
        # per-event sustained-recovery marker, if identifiable (Figure: event-level recovery)
        key_base = f"event_TTR_{e['event_type']}"
        if m.get(f"{key_base}_identifiable"):
            t_r_k = float(e["start_min"]) + m[f"{key_base}_min"]
            ax.axvline(t_r_k / 60.0, color="tab:green", linewidth=1, linestyle=":", alpha=.7)
    ax.axvline(m["t_e_min"] / 60.0, color="tab:red", linewidth=2, label=r"$t_e$ (first disruption onset)")
    ax.scatter([m["t_d_first_to_final_min"] / 60.0], [m["phi_d_first_to_final"]], color="black", zorder=5, label=r"$t_d$ (max degradation)")
    if m["recovered_within_horizon"]:
        t_r = m["t_r_first_to_final_min"]
        ax.axvline(t_r / 60.0, color="tab:green", linewidth=2, label=r"$t_r$ (first-to-final sustained recovery)")
        ax.plot([m["t_d_first_to_final_min"]/60.0, t_r/60.0], [m["phi_d_first_to_final"], values[int(round(t_r/config.DT_MIN))]],
                color="purple", linestyle="--", linewidth=1.5, label=r"avg. recovery rate (slope, $t_d \to t_r$, first-to-final)")
    else:
        ax.text(0.98, 0.05, "Not recovered within horizon (first-to-final TTR censored)", transform=ax.transAxes,
                ha="right", color="tab:red", fontsize=9)
    ax.plot([], [], color="tab:green", linewidth=1, linestyle=":", label="per-event sustained recovery (dotted)")

    ax.set_xlabel("Time [h]"); ax.set_ylabel("Throughput [cartons/h]")
    ax.set_title(title)
    subtitle = (
        f"R_phi(tf)={m['R_phi_tf']:.3f} [OPTIMIZED] | PL={m['PL_cartons']:.2f} cartons | TST={m['TST_min']:.1f} min | TUT={m['TUT_min']:.1f} min | "
        + (f"TTR(first-to-final)={m['TTR_first_to_final_min']:.1f} min" if m["recovered_within_horizon"] else "TTR(first-to-final)=censored")
        + f" | total_recovery_interval={m['total_recovery_interval_min']:.1f} min"
        + (f" | TTR(event mean)={m['TTR_event_mean_min']:.1f} min" if not np.isnan(m["TTR_event_mean_min"]) else " | TTR(event mean)=unresolved")
        + (f" | Average normalized recovery rate (first-to-final) [complementary]={m['recovery_rate_norm_per_h_first_to_final']:.3f} h^-1" if not np.isnan(m["recovery_rate_norm_per_h_first_to_final"]) else "")
    )
    ax.text(0.5, 1.12, subtitle, transform=ax.transAxes, ha="center", fontsize=8)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)

def scenario_comparison(rows: Sequence[Mapping], path: Path, title: str,
                         metrics=(("R_phi_tf_mean","R_phi_tf_std","R_phi(tf) [-] [OPTIMIZED]"),
                                   ("total_recovery_interval_mean_min","total_recovery_interval_std_min","Total recovery interval [min]"),
                                   ("TTR_first_to_final_physical_mean_min","TTR_first_to_final_physical_std_min","TTR first-to-final [min]"),
                                   ("event_recovery_rate_norm_per_h_realization_mean","event_recovery_rate_norm_per_h_realization_std","Average normalized recovery rate (realization-weighted episode mean) [h^-1]"),
                                   ("PL_cartons_mean","PL_cartons_std","PL [cartons]"),
                                   ("TST_min_mean","TST_min_std","TST [min]"),
                                   ("TUT_min_mean","TUT_min_std","TUT [min]"))):
    """Scenario-wise comparison of resilience metrics (mean +/- std over the LHS realisations), one subplot per metric.
        rows: one dict per scenario with the mean/std keys named in metrics.
    """
    names = [r["scenario_name"] for r in rows]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 5.5))
    if len(metrics) == 1:
        axes = [axes]
    for ax, (mean_key, std_key, label) in zip(axes, metrics):
        means = [r.get(mean_key, float("nan")) for r in rows]
        stds = [r.get(std_key, 0.0) for r in rows]
        ax.bar(range(len(rows)), means, yerr=stds, capsize=4, color="tab:blue", alpha=.75)
        ax.set_xticks(range(len(rows))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(label)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
