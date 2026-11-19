from __future__ import annotations
from . import config, mtbf_mttr, scenario_resilience_study
from .analysis import evaluate_umda, mean_time_profile, uv_sensitivity
from .design import DV_ORDER
from .exporters import export_workbook, write_csv
from .optimisation import display_label, refine_pareto_front, representatives, run_nsga2
from . import run_safety
from .plots import (convergence_diagnostics, cumulative_production_time, pareto_3d, resilience_time,
                     resilience_time_normalized, resilience_trajectory_annotated, scenario_comparison)
from .des import build_events as _build_events
from .scenarios import get_scenario
from .sensitivity import dv_oat
from .uncertainty import UV_ORDER, distributions_for_design, lhs_unit

def _event_level_table(uncertainty_rows):
    """Flatten the per-event event_TTR_<type>_* fields of the uncertainty-propagation rows into a table with one row per
        episode and realisation (event-level TTR and recovery rate).
    """
    if not uncertainty_rows:
        return []
    # '_identifiable' is a unique suffix ('_min' is shared by several fields such as TTR_min, t_e_min and end_min).
    event_types = sorted({k[len("event_TTR_"):-len("_identifiable")] for k in uncertainty_rows[0]
                           if k.startswith("event_TTR_") and k.endswith("_identifiable")})
    rows = []
    for r in uncertainty_rows:
        for et in event_types:
            base = f"event_TTR_{et}"
            rows.append({
                "representative_solution": r["representative_solution"],
                "run_id": r["run_id"],
                "event_type": et,
                "t_e_min": r.get(f"{base}_t_e_min"),
                "end_min": r.get(f"{base}_end_min"),
                "t_d_min": r.get(f"{base}_t_d_min"),
                "t_r_min": r.get(f"{base}_t_r_min"),
                "TTR_min": r.get(f"{base}_min"),
                "recovery_rate_norm_per_h": r.get(f"{base}_recovery_rate_norm_per_h"),
                "recovery_rate_TH_per_h2": r.get(f"{base}_recovery_rate_TH_per_h2"),
                "status": r.get(f"{base}_status"),
                "identifiable": r.get(f"{base}_identifiable"),
                "part_of_compound_episode": r.get(f"{base}_part_of_compound_episode"),
                "compound_TTR_min": r.get(f"{base}_compound_TTR_min"),
                "compound_recovery_rate_norm_per_h": r.get(f"{base}_compound_recovery_rate_norm_per_h"),
                "compound_recovery_rate_TH_per_h2": r.get(f"{base}_compound_recovery_rate_TH_per_h2"),
            })
    return rows

def _shift_level_table(rep_rows):
    """Shift-level recovery indicators: one row per representative design (reported indicators, not objectives)."""
    return [{
        "representative_solution": r["representative_solution"],
        "representative_solution_label": display_label(r["representative_solution"]),
        "R_phi_tf_mean": r["R_phi_tf_mean"], "R_phi_end_shift_mean": r["R_phi_end_shift_mean"],
        "PL_cartons_mean": r["PL_cartons_mean"], "PL_cartons_std": r["PL_cartons_std"],
        "TST_min_mean": r["TST_min_mean"], "TST_min_std": r["TST_min_std"],
        "TUT_min_mean": r["TUT_min_mean"], "TUT_min_std": r["TUT_min_std"],
        "total_recovery_interval_mean_min": r["total_recovery_interval_mean_min"],
        "total_recovery_interval_std_min": r["total_recovery_interval_std_min"],
        "total_recovery_interval_censored_fraction": r["total_recovery_interval_censored_fraction"],
        "TTR_event_mean_physical_mean_min": r["TTR_event_mean_physical_mean_min"],
        "TTR_event_mean_physical_std_min": r["TTR_event_mean_physical_std_min"],
        "TTR_event_mean_unresolved_fraction": r["TTR_event_mean_unresolved_fraction"],
        "TTR_event_mean_n_recovered_total": r["TTR_event_mean_n_recovered_total"],
        "TTR_event_mean_n_overlapping_superseded_total": r["TTR_event_mean_n_overlapping_superseded_total"],
        "TTR_event_mean_n_not_recovered_total": r["TTR_event_mean_n_not_recovered_total"],
        "TTR_event_mean_n_not_identifiable_total": r["TTR_event_mean_n_not_identifiable_total"],
        "TTR_first_to_final_physical_mean_min": r["TTR_first_to_final_physical_mean_min"],
        "TTR_first_to_final_physical_std_min": r["TTR_first_to_final_physical_std_min"],
        "TTR_first_to_final_censored_fraction": r["TTR_first_to_final_censored_fraction"],
        # Average normalised recovery rate [1/h], two aggregations (see analysis.py): 'realization' weights each LHS
        # realisation equally (headline value); 'pooled' weights each episode equally (diagnostic).
        "R_rate_mean_h_inv": r["event_recovery_rate_norm_per_h_realization_mean"],
        "R_rate_std_h_inv": r["event_recovery_rate_norm_per_h_realization_std"],
        "R_rate_median_h_inv": r["event_recovery_rate_norm_per_h_realization_median"],
        "R_rate_q05_h_inv": r["event_recovery_rate_norm_per_h_realization_q05"],
        "R_rate_q95_h_inv": r["event_recovery_rate_norm_per_h_realization_q95"],
        "R_rate_n_realizations_valid": r["event_recovery_rate_n_realizations_valid"],
        "R_rate_n_realizations_undefined": r["event_recovery_rate_n_realizations_undefined"],
        "R_rate_realization_undefined_fraction": r["event_recovery_rate_realization_undefined_fraction"],
        "R_rate_pooled_mean_h_inv": r["event_recovery_rate_norm_per_h_pooled_mean"],
        "R_rate_pooled_std_h_inv": r["event_recovery_rate_norm_per_h_pooled_std"],
        "R_rate_pooled_median_h_inv": r["event_recovery_rate_norm_per_h_pooled_median"],
        "R_rate_pooled_q05_h_inv": r["event_recovery_rate_norm_per_h_pooled_q05"],
        "R_rate_pooled_q95_h_inv": r["event_recovery_rate_norm_per_h_pooled_q95"],
        "R_rate_pooled_n_valid": r["event_recovery_rate_pooled_n_valid"],
        # Absolute-unit counterparts (cartons/h^2); not used to compare designs, because the nominal throughput
        # differs.
        "R_rate_mean_cartons_per_h2": r["event_recovery_rate_TH_per_h2_realization_mean"],
        "R_rate_std_cartons_per_h2": r["event_recovery_rate_TH_per_h2_realization_std"],
        "R_rate_pooled_mean_cartons_per_h2": r["event_recovery_rate_TH_per_h2_pooled_mean"],
        "R_rate_pooled_std_cartons_per_h2": r["event_recovery_rate_TH_per_h2_pooled_std"],
        # First-to-final recovery rate: one slope per realisation from the global trough to the final recovery point
        # (see analysis.py).
        "recovery_rate_norm_per_h_first_to_final_mean": r["recovery_rate_norm_per_h_first_to_final_mean"],
        "recovery_rate_norm_per_h_first_to_final_std": r["recovery_rate_norm_per_h_first_to_final_std"],
        "recovery_rate_norm_per_h_first_to_final_n_valid": r["recovery_rate_norm_per_h_first_to_final_n_valid"],
        "P_recover_within_shift": r["P_recover_within_shift"],
    } for r in rep_rows]

def run(run_label: str = "exploratory_run"):
    """Run the complete UMDO workflow. The outputs are written to a dedicated directory
        results/<run_label>_<timestamp>/ (run_safety.make_run_dir), so that repeated runs never overwrite each other.
    """
    scenario=get_scenario(config.SCENARIO_MODE)
    out_dir = run_safety.make_run_dir(run_label)
    run_safety.write_config_snapshot(out_dir)
    print(f"[main.run] output directory for this run: {out_dir}")

    optimisation=run_nsga2(out_dir=out_dir)
    if optimisation.X is None:
        raise RuntimeError(
            "NSGA2 found no individuals at all (empty final population). "
            "This should not happen with return_least_infeasible=True unless "
            "the search itself failed to run - check NSGA2_POPULATION/GENERATIONS."
        )
    # Save the raw NSGA-II result before any further analysis (most expensive stage).
    raw_result_path = run_safety.save_nsga2_raw_result(out_dir, optimisation)
    print(f"[main.run] raw NSGA-II result saved: {raw_result_path}")
    # Per-generation convergence diagnostics of this run (optimisation.DiagnosticsCallback).
    convergence_diagnostics(
        optimisation.diagnostics_callback.rows, out_dir/"nsga2_convergence_diagnostics.png",
        f"NSGA-II convergence diagnostics - {scenario['name']}",
    )
    final_lhs=lhs_unit(config.N_LHS_FINAL,len(UV_ORDER),config.LHS_SEED+1000)

    # Two-stage feasibility: the search uses N_LHS_OPTIMISATION samples; every candidate is re-evaluated at
    # N_LHS_FINAL, candidates violating any of the nine constraints are removed and non-dominance is recomputed. All
    # re-evaluated candidates are kept in pareto_candidates_all.
    pareto,pareto_candidates_all,stage_counts=refine_pareto_front(optimisation.X,final_lhs)
    print(f"Pipeline stage counts: {stage_counts}")
    if stage_counts["n_unique"] < stage_counts["n_evaluated_raw"]:
        print(f"  (deduplicated {stage_counts['n_evaluated_raw']-stage_counts['n_unique']} "
              f"NSGA-II individuals that rounded to an already-seen integer design)")
    if not pareto:
        raise RuntimeError(
            "No candidate survives the final high-fidelity (N_LHS_FINAL) "
            "re-confirmation of all 9 constraints, even though NSGA2 found "
            "some feasible-looking candidates during the coarser search. "
            "Consider whether config.RELIABILITY_LEVEL or the deterministic "
            "engineering thresholds are too restrictive for the current "
            "design space, or increase NSGA2_POPULATION/GENERATIONS."
        )

    reps=representatives(pareto)
    rep_rows=[]; uncertainty=[]; uv_sens=[]; time_rows=[]
    representative_designs=[]

    def add_representative(name,design,uv_design=None):
        representative_designs.append((name,design,uv_design))
        summary,runs,profiles=evaluate_umda(design,final_lhs,keep_profiles=True,uv_design=uv_design)
        rep_rows.append({"representative_solution":name,**summary})
        uncertainty.extend({"representative_solution":name,**r} for r in runs)
        uv_sens.extend({"representative_solution":name,**r} for r in uv_sensitivity(runs))
        time_rows.extend({"representative_solution":name,**r} for r in mean_time_profile(profiles))

    for name,row in reps.items():
        add_representative(name,{dv:int(row[dv]) for dv in DV_ORDER})

    # not_improv (reference design): the max-resilience design with the disruption-duration distributions reset to
    # their baseline means (all-level-1 uv_design, coefficient 1.0 for every uncertain variable). It isolates the
    # effect of the design-dependent shortening of the disruption durations.
    max_resilience_design={dv:int(reps["max_resilience"][dv]) for dv in DV_ORDER}
    add_representative("not_improv",max_resilience_design,uv_design={dv:1 for dv in DV_ORDER})

    nominal_th_by_rep={r["representative_solution"]:float(r["nominal_throughput"]) for r in rep_rows}
    time_rows_normalized=[
        {**r,"throughput_normalized":r["throughput_mean"]/nominal_th_by_rep[r["representative_solution"]]}
        for r in time_rows
    ]
    # Undisrupted reference: cumulative cartons at the design's own nominal rate. The gap to the disrupted curve is
    # the production lost to the disruptions (no catch-up is modelled).
    time_rows_cumulative=[
        {**r,"cumulative_cartons_reference":nominal_th_by_rep[r["representative_solution"]]*r["time_h"]}
        for r in time_rows
    ]

    # Additional run with the shift start-up ramp switched on (throughput ramps up from 0 at t = 0, see des.simulate),
    # kept separate from the default results.
    original_startup_ramp=config.MODEL_STARTUP_RAMP
    config.MODEL_STARTUP_RAMP=True
    try:
        realistic_time_rows=[]
        for name,design,uv_design in representative_designs:
            _,_,profiles=evaluate_umda(design,final_lhs,keep_profiles=True,uv_design=uv_design)
            realistic_time_rows.extend({"representative_solution":name,**r} for r in mean_time_profile(profiles))
    finally:
        config.MODEL_STARTUP_RAMP=original_startup_ramp

    balanced={dv:int(reps["middle_balanced"][dv]) for dv in DV_ORDER}
    dv_rows=dv_oat(balanced)
    scenario_rows=[{"scenario_mode":config.SCENARIO_MODE,"scenario_name":scenario["name"],"event_type":e,"start_h":t} for e,t in scenario["events"]]

    important=[{
        "representative_solution":r["representative_solution"],
        "representative_solution_label":display_label(r["representative_solution"]),
        **{dv:r[dv] for dv in DV_ORDER},
        "TH_nominal":r["nominal_throughput"],
        "TH_disrupted_mean":r["disrupted_throughput_mean"],
        "TH_disrupted_std":r["disrupted_throughput_std"],
        "Energy_mean_kWh":r["energy_kWh_mean"],
        "Energy_std_kWh":r["energy_kWh_std"],
        "R_phi_tf_mean":r["R_phi_tf_mean"],
        "R_phi_end_shift_mean":r["R_phi_end_shift_mean"],
        "PL_mean_cartons":r["PL_cartons_mean"],
        "TST_mean_min":r["TST_min_mean"],
        "TUT_mean_min":r["TUT_min_mean"],
        # TTR family: reported indicators, not objectives.
        "total_recovery_interval_mean_min":r["total_recovery_interval_mean_min"],
        "total_recovery_interval_std_min":r["total_recovery_interval_std_min"],
        "total_recovery_interval_censored_fraction":r["total_recovery_interval_censored_fraction"],
        "TTR_event_mean_physical_mean_min":r["TTR_event_mean_physical_mean_min"],
        "TTR_event_mean_physical_std_min":r["TTR_event_mean_physical_std_min"],
        "TTR_event_mean_unresolved_fraction":r["TTR_event_mean_unresolved_fraction"],
        "TTR_first_to_final_physical_mean_min":r["TTR_first_to_final_physical_mean_min"],
        "TTR_first_to_final_physical_std_min":r["TTR_first_to_final_physical_std_min"],
        "TTR_first_to_final_censored_fraction":r["TTR_first_to_final_censored_fraction"],
        "P_recover_TTR_first_to_final":r["P_recover_TTR_first_to_final"],
        "Average normalized recovery rate [h^-1]":r["event_recovery_rate_norm_per_h_realization_mean"],
        "Average normalized recovery rate std [h^-1]":r["event_recovery_rate_norm_per_h_realization_std"],
        "R_rate_n_realizations_valid":r["event_recovery_rate_n_realizations_valid"],
        "R_rate_realization_undefined_fraction":r["event_recovery_rate_realization_undefined_fraction"],
        "event_recovery_rate_norm_per_h_pooled_mean":r["event_recovery_rate_norm_per_h_pooled_mean"],
        "event_recovery_rate_norm_per_h_pooled_std":r["event_recovery_rate_norm_per_h_pooled_std"],
        "recovery_rate_norm_per_h_first_to_final_mean":r["recovery_rate_norm_per_h_first_to_final_mean"],
        "recovery_rate_norm_per_h_first_to_final_std":r["recovery_rate_norm_per_h_first_to_final_std"],
        "recovery_rate_norm_per_h_first_to_final_n_valid":r["recovery_rate_norm_per_h_first_to_final_n_valid"],
        "starvation_mean_min":r["starvation_min_mean"],
        "blockage_mean_min":r["blockage_min_mean"],
        "P_recover_within_shift":r["P_recover_within_shift"],
        "probability_TH_requirement_satisfied":r["probability_TH_requirement_satisfied"],
        "probability_energy_requirement_satisfied":r["probability_energy_requirement_satisfied"],
        "g_TH_det":r["g_TH_det"], "g_WIP_det":r["g_WIP_det"], "g_CT_det":r["g_CT_det"],
        "g_E_det":r["g_E_det"], "g_e_carton_det":r["g_e_carton_det"], "g_MDA":r["g_MDA"],
        "g_TH_prob":r["g_TH_prob"], "g_E_prob":r["g_E_prob"], "g_recovery":r["g_recovery"],
        "feasible_det":r["feasible_det"], "feasible_uncertain":r["feasible_uncertain"],
        "feasible_overall":r["feasible_overall"], "total_constraint_violation":r["total_constraint_violation"],
    } for r in rep_rows]

    event_level_rows = _event_level_table(uncertainty)
    shift_level_rows = _shift_level_table(rep_rows)

    # Output file names carry the scenario suffix.
    suffix=f"scenario_{config.SCENARIO_MODE}_{scenario['name'].lower()}"
    out=out_dir  # this run's own dedicated subdirectory - see make_run_dir() above
    write_csv(out/f"pipeline_stage_counts_{suffix}.csv", [stage_counts])

    # MTBF/MTTR for the multi-event scenarios 4-7 (independent of config.SCENARIO_MODE), for the balanced
    # representative design.
    mtbf_mttr_rows = mtbf_mttr.run(balanced, modes=(4, 5, 6, 7))
    write_csv(out/"mtbf_mttr_scenarios_4_7.csv", mtbf_mttr_rows)

    # Scenario-wise resilience comparison over all seven scenarios for the balanced design (independent of
    # config.SCENARIO_MODE).
    scenario_comparison_rows = scenario_resilience_study.run(balanced, modes=(1,2,3,4,5,6,7))
    write_csv(out/"scenario_resilience_comparison.csv", scenario_comparison_rows)
    scenario_comparison(
        scenario_comparison_rows,
        out/"scenario_resilience_comparison.png",
        "Resilience metrics by disruption scenario (balanced design)",
    )
    write_csv(out/f"pareto_front_{suffix}.csv",pareto)
    write_csv(out/f"pareto_candidates_all_{suffix}.csv",pareto_candidates_all)
    write_csv(out/f"uncertainty_propagation_{suffix}.csv",uncertainty)
    write_csv(out/f"uv_sensitivity_lhs_{suffix}.csv",uv_sens)
    write_csv(out/f"dv_oat_lhs_{suffix}.csv",dv_rows)
    write_csv(out/f"resilience_time_profile_{suffix}.csv",time_rows)
    write_csv(out/f"resilience_time_profile_normalized_{suffix}.csv",time_rows_normalized)
    write_csv(out/f"resilience_time_profile_realistic_{suffix}.csv",realistic_time_rows)
    write_csv(out/f"cumulative_production_time_profile_{suffix}.csv",time_rows_cumulative)
    write_csv(out/f"event_level_recovery_{suffix}.csv",event_level_rows)
    write_csv(out/f"shift_level_recovery_{suffix}.csv",shift_level_rows)

    export_workbook(out/f"important_results_{suffix}.xlsx",{
        "Important_Metrics":important,
        "Scenario":scenario_rows,
        "MDO":[{k:v for k,v in r.items() if k in ["representative_solution",*DV_ORDER,"nominal_throughput","nominal_energy_kWh","cycle_time_s","mda_residual","mda_converged"]} for r in rep_rows],
        "UMDA":rep_rows,
        "UMDO_Pareto":pareto,
        "Pareto_Candidates_All":pareto_candidates_all,
        "Representatives":rep_rows,
        "UV_Sensitivity_LHS":uv_sens,
        "DV_OAT_LHS":dv_rows,
        "Uncertainty_Runs":uncertainty,
        "Time_Profile":time_rows,
        "Cumulative_Production":time_rows_cumulative,
        "MTBF_MTTR_Scenarios_4_7":mtbf_mttr_rows,
        "Scenario_Resilience_Comparison":scenario_comparison_rows,
        "Event_Level_Recovery":event_level_rows,
        "Shift_Level_Recovery":shift_level_rows,
    })
    export_workbook(out/f"pareto_front_{suffix}.xlsx",{"Pareto_Front":pareto,"Pareto_Candidates_All":pareto_candidates_all,"Representatives":rep_rows})
    export_workbook(out/f"resilience_over_time_{suffix}.xlsx",{"Resilience_Over_Time":time_rows})

    # 3D Pareto front with the R_phi(t_f) axis (maximised); physical values are plotted.
    pareto_3d(pareto,reps,out/f"pareto_3d_{suffix}.png",f"3D Pareto front (R_phi) - {scenario['name']}")
    resilience_time(time_rows,out/f"resilience_over_time_{suffix}.png",f"Resilience over time - {scenario['name']}")
    resilience_time_normalized(time_rows_normalized,out/f"resilience_over_time_normalized_{suffix}.png",f"Resilience performance over time - {scenario['name']}")

    only_resilience_and_not_improv=[r for r in time_rows_normalized if r["representative_solution"] in ("max_resilience","not_improv")]
    resilience_time_normalized(
        only_resilience_and_not_improv,
        out/f"resilience_over_time_normalized_only_resilience_and_not_improv_{suffix}.png",
        f"Normalized performance over time: {display_label('max_resilience')} vs {display_label('not_improv')} - {scenario['name']}",
    )

    resilience_time(
        realistic_time_rows,
        out/f"resilience_realistic_behaviour_over_time_{suffix}.png",
        f"Resilience over time (realistic startup ramp) - {scenario['name']}",
    )

    cumulative_production_time(
        time_rows_cumulative,
        out/f"cumulative_production_over_time_{suffix}.png",
        f"Cumulative production over time - {scenario['name']}",
    )

    # Annotated resilience trajectory of the max-resilience representative; the events use the design's mean
    # disruption durations (annotation only).
    max_res_design={dv:int(reps["max_resilience"][dv]) for dv in DV_ORDER}
    mean_sample={uv:dist.mean for uv,dist in distributions_for_design(max_res_design).items()}
    annotated_events=_build_events(mean_sample)
    max_res_time_rows=[r for r in time_rows if r["representative_solution"]=="max_resilience"]
    resilience_trajectory_annotated(
        max_res_time_rows, annotated_events, nominal_th_by_rep["max_resilience"],
        out/f"resilience_trajectory_annotated_{suffix}.png",
        f"Annotated resilience trajectory ({display_label('max_resilience')}) - {scenario['name']}",
    )

    print("="*80)
    print("UMDO completed")
    print(f"Scenario: {config.SCENARIO_MODE} - {scenario['name']}")
    print("Resilience optimization objective: R_phi(tf) (sole objective; TTR/recovery rate/PL/TST/TUT are complementary, non-optimized indicators)")
    print(f"Pareto solutions: {len(pareto)}")
    print(f"Outputs: {out.resolve()}")
    for r in rep_rows:
        print(
            f"{display_label(r['representative_solution'])}: "
            f"R_phi(tf)={r['R_phi_tf_mean']:.4f} [OPTIMIZED], "
            f"TH={r['disrupted_throughput_mean']:.4f}, E={r['energy_kWh_mean']:.4f}, "
            f"PL={r['PL_cartons_mean']:.2f} cartons, TST={r['TST_min_mean']:.2f} min, TUT={r['TUT_min_mean']:.2f} min, "
            f"TTR(first-to-final)={r['TTR_first_to_final_physical_mean_min']:.2f} min "
            f"(censored {r['TTR_first_to_final_censored_fraction']*100:.1f}%), "
            f"total_recovery_interval={r['total_recovery_interval_mean_min']:.2f}+/-{r['total_recovery_interval_std_min']:.2f} min, "
            f"Average normalized recovery rate={r['event_recovery_rate_norm_per_h_realization_mean']:.4f} h^-1 "
            f"(n={r['event_recovery_rate_n_realizations_valid']} valid realizations)"
        )

if __name__ == "__main__":
    run()
