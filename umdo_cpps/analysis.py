from __future__ import annotations
from typing import Any, Dict, Mapping, Sequence
import numpy as np
from . import config
from .deterministic_adapter import evaluate_mdo, thresholds as deterministic_thresholds
from .des import simulate
from .metrics import calculate
from .uncertainty import UV_ORDER, lhs_unit, samples_for_design

def evaluate_umda(design: Mapping[str, int], unit_lhs: np.ndarray, keep_profiles: bool = False, uv_design: Mapping[str, int] | None = None):
    mdo = evaluate_mdo(design)
    samples = samples_for_design(uv_design if uv_design is not None else design, unit_lhs)
    rows = []
    profiles = []
    for run_id, sample in enumerate(samples, 1):
        run = simulate(design, mdo, sample, keep_profile=True)
        metrics = calculate(run["profile"], run["events"], float(mdo["nominal_throughput"]))
        row = {
            "run_id": run_id, **sample,
            **{k:v for k,v in run.items() if k not in ("profile","events")},
            **metrics,
        }
        rows.append(row)
        if keep_profiles:
            profiles.append(run["profile"])

    summary: Dict[str, Any] = {**dict(design), **mdo}
    keys = ["disrupted_throughput","energy_kWh","PL_cartons","R_phi_tf","R_phi_end_shift","TST_min","TUT_min","starvation_min","blockage_min"]
    for key in keys:
        values = np.asarray([float(r[key]) for r in rows])
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values)>1 else 0.0

    # Reliability statistics over the LHS realisations.
    disrupted_throughput_vals = np.asarray([float(r["disrupted_throughput"]) for r in rows])
    disrupted_energy_vals = np.asarray([float(r["energy_kWh"]) for r in rows])
    tst_vals = np.asarray([float(r["TST_min"]) for r in rows])

    th = deterministic_thresholds()

    summary["disrupted_throughput_q05"] = float(np.percentile(disrupted_throughput_vals, 5))
    summary["disrupted_throughput_q50"] = float(np.percentile(disrupted_throughput_vals, 50))
    summary["disrupted_throughput_q95"] = float(np.percentile(disrupted_throughput_vals, 95))
    summary["disrupted_energy_q05"] = float(np.percentile(disrupted_energy_vals, 5))
    summary["disrupted_energy_q50"] = float(np.percentile(disrupted_energy_vals, 50))
    summary["disrupted_energy_q95"] = float(np.percentile(disrupted_energy_vals, 95))

    summary["probability_TH_requirement_satisfied"] = float(np.mean(disrupted_throughput_vals >= th["THROUGHPUT_MIN"]))
    summary["probability_energy_requirement_satisfied"] = float(np.mean(disrupted_energy_vals <= th["E_TOT_MAX"]))
    # TST = SHIFT_HOURS * 60 means 'no sustained recovery within the shift' (see metrics.calculate()), hence the
    # strict inequality.
    summary["P_recover_within_shift"] = float(np.mean(tst_vals < config.SHIFT_HOURS * 60.0))

    # Complementary TTR and recovery-rate indicators (reported only; R_phi(t_f) is the only resilience objective).
    n_total = len(rows)

    # --- First-to-final recovery span (can include normal operation between disruptions; see total_recovery_interval)
    # ---
    ftf_censored_mask = np.asarray([bool(r["TTR_first_to_final_censored"]) for r in rows])
    summary["TTR_first_to_final_censored_count"] = int(np.sum(ftf_censored_mask))
    summary["TTR_first_to_final_censored_fraction"] = float(np.mean(ftf_censored_mask)) if n_total else 0.0
    summary["P_recover_TTR_first_to_final"] = 1.0 - summary["TTR_first_to_final_censored_fraction"]

    physical_ftf = np.asarray([float(r["TTR_first_to_final_min"]) for r in rows])[~ftf_censored_mask]
    summary["TTR_first_to_final_physical_mean_min"] = float(np.mean(physical_ftf)) if physical_ftf.size else float("nan")
    summary["TTR_first_to_final_physical_std_min"] = float(np.std(physical_ftf, ddof=1)) if physical_ftf.size > 1 else 0.0
    for q in (5, 50, 95):
        summary[f"TTR_first_to_final_physical_q{q:02d}_min"] = float(np.percentile(physical_ftf, q)) if physical_ftf.size else float("nan")

    # first_to_final is one slope per realisation, from the global trough of the whole shift to the recovery point of
    # the final episode. It is not the mean of the per-episode rates (event_recovery_rate_norm_per_h_realization_*)
    # and is reported as a diagnostic only.
    for key in ("recovery_rate_norm_per_h_first_to_final", "recovery_rate_TH_per_h2_first_to_final"):
        vals = np.asarray([float(r[key]) for r in rows])
        valid = vals[~np.isnan(vals)]
        summary[f"{key}_mean"] = float(np.mean(valid)) if valid.size else float("nan")
        summary[f"{key}_std"] = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
        summary[f"{key}_n_valid"] = int(valid.size)
        summary[f"{key}_n_undefined"] = int(vals.size - valid.size)

    # --- Total recovery-interval duration (union of the episode intervals, excluding normal operation between
    # completed recoveries; not an MTTR, censoring is flagged separately) ---
    total_interval_vals = np.asarray([float(r["total_recovery_interval_min"]) for r in rows])
    total_interval_censored_mask = np.asarray([bool(r["total_recovery_interval_censored"]) for r in rows])
    summary["total_recovery_interval_mean_min"] = float(np.mean(total_interval_vals))
    summary["total_recovery_interval_std_min"] = float(np.std(total_interval_vals, ddof=1)) if total_interval_vals.size > 1 else 0.0
    for q in (5, 50, 95):
        summary[f"total_recovery_interval_q{q:02d}_min"] = float(np.percentile(total_interval_vals, q))
    summary["total_recovery_interval_censored_fraction"] = float(np.mean(total_interval_censored_mask))

    # --- Mean event-level TTR (identifiable, recovered episodes) and episode-status counts over the N realisations
    # ---
    event_mean_vals = np.asarray([float(r["TTR_event_mean_min"]) for r in rows])
    event_mean_nan_mask = np.isnan(event_mean_vals)
    physical_em = event_mean_vals[~event_mean_nan_mask]
    summary["TTR_event_mean_physical_mean_min"] = float(np.mean(physical_em)) if physical_em.size else float("nan")
    summary["TTR_event_mean_physical_std_min"] = float(np.std(physical_em, ddof=1)) if physical_em.size > 1 else 0.0
    summary["TTR_event_mean_unresolved_fraction"] = float(np.mean(event_mean_nan_mask)) if n_total else 0.0
    for status_key in ("TTR_event_mean_n_recovered", "TTR_event_mean_n_overlapping_superseded",
                       "TTR_event_mean_n_not_recovered", "TTR_event_mean_n_not_identifiable"):
        summary[f"{status_key}_total"] = int(sum(int(r[status_key]) for r in rows))

    # Average recovery rate (R_rate). metrics.calculate() computes it per episode as (phi(t_r) - phi(t_d)) / (t_r -
    # t_d), from the maximum-degradation point to the sustained-recovery point of the same episode, in hours. Two
    # scenario-level aggregates are reported separately:
    # A) pooled (episode-weighted): every identifiable episode of every realisation counts once; grouped by event type
    # (CNC_BREAKDOWN, COBOT_BREAKDOWN, ASSEMBLY_ERROR), so compound episodes are excluded.
    # B) realization (realisation-weighted): every realisation contributes its own mean over its episodes, including
    # compound episodes.
    # Event types are identified by the unique '_identifiable' suffix ('..._compound_recovery_rate_norm_per_h' also
    # ends with '_recovery_rate_norm_per_h').
    event_type_keys = sorted({k[len("event_TTR_"):-len("_identifiable")]
                               for k in rows[0] if k.startswith("event_TTR_") and k.endswith("_identifiable")})
    # event_recovery_rate_norm_per_h_realization_mean is the main R_rate indicator for comparing designs; the pooled
    # and first-to-final variants are diagnostics.
    pooled_rates = []
    pooled_rates_nonnorm = []
    for et in event_type_keys:
        vals = np.asarray([float(r[f"event_TTR_{et}_recovery_rate_norm_per_h"]) for r in rows])
        vals_nonnorm = np.asarray([float(r[f"event_TTR_{et}_recovery_rate_TH_per_h2"]) for r in rows])
        valid = vals[~np.isnan(vals)]
        valid_nonnorm = vals_nonnorm[~np.isnan(vals_nonnorm)]
        pooled_rates.extend(valid.tolist())
        pooled_rates_nonnorm.extend(valid_nonnorm.tolist())
        summary[f"event_recovery_rate_norm_per_h_{et}_mean"] = float(np.mean(valid)) if valid.size else float("nan")
        summary[f"event_recovery_rate_norm_per_h_{et}_std"] = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
        summary[f"event_recovery_rate_norm_per_h_{et}_n_valid"] = int(valid.size)
    summary["event_recovery_rate_norm_per_h_pooled_mean"] = float(np.mean(pooled_rates)) if pooled_rates else float("nan")
    summary["event_recovery_rate_norm_per_h_pooled_std"] = float(np.std(pooled_rates, ddof=1)) if len(pooled_rates) > 1 else 0.0
    summary["event_recovery_rate_norm_per_h_pooled_median"] = float(np.median(pooled_rates)) if pooled_rates else float("nan")
    for q in (5, 95):
        summary[f"event_recovery_rate_norm_per_h_pooled_q{q:02d}"] = float(np.percentile(pooled_rates, q)) if pooled_rates else float("nan")
    summary["event_recovery_rate_pooled_n_valid"] = len(pooled_rates)
    # Absolute-units (cartons/h^2) counterpart of the episode-pooled aggregate.
    summary["event_recovery_rate_TH_per_h2_pooled_mean"] = float(np.mean(pooled_rates_nonnorm)) if pooled_rates_nonnorm else float("nan")
    summary["event_recovery_rate_TH_per_h2_pooled_std"] = float(np.std(pooled_rates_nonnorm, ddof=1)) if len(pooled_rates_nonnorm) > 1 else 0.0
    summary["event_recovery_rate_TH_per_h2_pooled_n_valid"] = len(pooled_rates_nonnorm)

    realization_rate_vals = np.asarray([float(r["episode_recovery_rate_mean_norm_per_h"]) for r in rows])
    realization_rate_nan_mask = np.isnan(realization_rate_vals)
    valid_realization_rates = realization_rate_vals[~realization_rate_nan_mask]
    summary["event_recovery_rate_norm_per_h_realization_mean"] = float(np.mean(valid_realization_rates)) if valid_realization_rates.size else float("nan")
    summary["event_recovery_rate_norm_per_h_realization_std"] = float(np.std(valid_realization_rates, ddof=1)) if valid_realization_rates.size > 1 else 0.0
    summary["event_recovery_rate_norm_per_h_realization_median"] = float(np.median(valid_realization_rates)) if valid_realization_rates.size else float("nan")
    for q in (5, 95):
        summary[f"event_recovery_rate_norm_per_h_realization_q{q:02d}"] = float(np.percentile(valid_realization_rates, q)) if valid_realization_rates.size else float("nan")
    summary["event_recovery_rate_n_realizations_valid"] = int(valid_realization_rates.size)
    summary["event_recovery_rate_n_realizations_undefined"] = int(np.sum(realization_rate_nan_mask))
    summary["event_recovery_rate_realization_undefined_fraction"] = float(np.mean(realization_rate_nan_mask)) if n_total else 0.0

    # Absolute-unit (cartons/h^2) counterpart of the realisation-weighted aggregate; it is undefined in the same
    # realisations as the normalised value.
    realization_rate_vals_nonnorm = np.asarray([float(r["episode_recovery_rate_mean_TH_per_h2"]) for r in rows])
    valid_realization_rates_nonnorm = realization_rate_vals_nonnorm[~realization_rate_nan_mask]
    summary["event_recovery_rate_TH_per_h2_realization_mean"] = float(np.mean(valid_realization_rates_nonnorm)) if valid_realization_rates_nonnorm.size else float("nan")
    summary["event_recovery_rate_TH_per_h2_realization_std"] = float(np.std(valid_realization_rates_nonnorm, ddof=1)) if valid_realization_rates_nonnorm.size > 1 else 0.0
    summary["event_recovery_rate_TH_per_h2_realization_n_valid"] = int(valid_realization_rates_nonnorm.size)

    # Constrained-UMDO constraint vector, pymoo convention g(x) <= 0 feasible.
    # Deterministic/nominal engineering feasibility (layers 2-3):
    summary["g_TH_det"] = th["THROUGHPUT_MIN"] - summary["nominal_throughput"]
    summary["g_WIP_det"] = summary["nominal_WIP"] - th["WIP_MAX"]
    summary["g_CT_det"] = summary["cycle_time_s"] - th["CT_MAX"]
    summary["g_E_det"] = summary["nominal_energy_kWh"] - th["E_TOT_MAX"]
    summary["g_e_carton_det"] = summary["nominal_energy_per_carton"] - th["E_CARTON_MAX"]
    summary["g_MDA"] = summary["mda_residual"] - th["MDA_TOL"]
    # Uncertainty/disruption reliability feasibility (layer 4):
    summary["g_TH_prob"] = th["THROUGHPUT_MIN"] - summary["disrupted_throughput_q05"]
    summary["g_E_prob"] = summary["disrupted_energy_q95"] - th["E_TOT_MAX"]
    summary["g_recovery"] = config.RELIABILITY_LEVEL - summary["P_recover_within_shift"]

    det_g_keys = ["g_TH_det","g_WIP_det","g_CT_det","g_E_det","g_e_carton_det","g_MDA"]
    uncertain_g_keys = ["g_TH_prob","g_E_prob","g_recovery"]
    all_g_keys = det_g_keys + uncertain_g_keys
    summary["feasible_det"] = bool(all(summary[k] <= 0.0 for k in det_g_keys))
    summary["feasible_uncertain"] = bool(all(summary[k] <= 0.0 for k in uncertain_g_keys))
    summary["feasible_overall"] = bool(summary["feasible_det"] and summary["feasible_uncertain"])
    summary["total_constraint_violation"] = float(sum(max(0.0, summary[k]) for k in all_g_keys))

    return summary, rows, profiles

def mean_time_profile(profiles: Sequence[Sequence[Mapping[str, Any]]]):
    if not profiles:
        return []
    rows = []
    for i in range(len(profiles[0])):
        rows.append({
            "time_min": profiles[0][i]["time_min"],
            "time_h": profiles[0][i]["time_h"],
            "throughput_mean": float(np.mean([p[i]["throughput_cartons_h"] for p in profiles])),
            "throughput_std": float(np.std([p[i]["throughput_cartons_h"] for p in profiles], ddof=1)),
            "buffer_CNC_Assembly_mean": float(np.mean([p[i]["buffer_CNC_Assembly"] for p in profiles])),
            "buffer_Assembly_QC_mean": float(np.mean([p[i]["buffer_Assembly_QC"] for p in profiles])),
            "energy_mean_kWh": float(np.mean([p[i]["cumulative_energy_kWh"] for p in profiles])),
            "cumulative_cartons_mean": float(np.mean([p[i]["cumulative_cartons"] for p in profiles])),
        })
    return rows

def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if np.std(x)<=1e-12 or np.std(y)<=1e-12:
        return 0.0
    return float(np.corrcoef(x,y)[0,1])

def src_coefficients(x, y):
    x = np.asarray(x, float); y=np.asarray(y,float)
    xs=(x-np.mean(x,axis=0))/np.where(np.std(x,axis=0)<=1e-12,1,np.std(x,axis=0))
    ys=(y-np.mean(y))/(np.std(y) if np.std(y)>1e-12 else 1)
    beta,*_=np.linalg.lstsq(xs,ys,rcond=None)
    return beta

def uv_sensitivity(run_rows: Sequence[Mapping[str, Any]]):
    outputs=["disrupted_throughput","energy_kWh","PL_cartons","R_phi_tf","TST_min","TUT_min"]
    x=np.asarray([[float(r[uv]) for uv in UV_ORDER] for r in run_rows])
    result=[]
    for out in outputs:
        y=np.asarray([float(r[out]) for r in run_rows])
        beta=src_coefficients(x,y)
        for i,uv in enumerate(UV_ORDER):
            result.append({
                "output":out,"uncertain_variable":uv,
                "pearson_correlation":pearson(x[:,i],y),
                "standardised_regression_coefficient":float(beta[i]),
                "absolute_SRC":abs(float(beta[i])),
            })
    return result
