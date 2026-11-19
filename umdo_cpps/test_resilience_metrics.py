"""Tests of the resilience metrics: TTR (first-to-final, total recovery interval, event mean), episode-level recovery
(including the merging of overlapping disruptions into compound episodes), recovery rate and the UMDO formulation
with R_phi(t_f) as the resilience objective.

    python -m umdo_cpps.test_resilience_metrics
"""
from __future__ import annotations
import math
import numpy as np
from . import config
from .metrics import calculate

def _profile(t0: int, t1: int, values):
    return [{"time_min": float(t), "throughput_cartons_h": float(v)} for t, v in zip(range(t0, t1), values)]

def test_A_ttr_tst_transient_recoveryrate():
    """Checks 5, 9: TTR from onset (not repair end), != TST; a transient
    crossing does not count as sustained recovery; recovery rate uses the
    t_d->t_r interval."""
    nominal_th = 10.0
    events = [{"event_type": "X", "start_min": 10.0, "end_min": 20.0}]
    values = []
    values += [10.0] * 10
    values += [2.0] * 10
    values += [2.0 + (t - 20) * (9.5 - 2.0) / 9 for t in range(20, 30)]
    values += [10.0]
    values += [5.0]
    values += [5.0 + (t - 32) * (10.0 - 5.0) / 7 for t in range(32, 40)]
    values += [10.0] * 20

    profile = _profile(0, 60, values)
    original_shift_hours = config.SHIFT_HOURS
    config.SHIFT_HOURS = 60.0 / 60.0
    try:
        m = calculate(profile, events, nominal_th)
    finally:
        config.SHIFT_HOURS = original_shift_hours

    assert m["recovered_within_horizon"] is True
    assert abs(m["t_r_first_to_final_min"] - 39.0) < 1e-6, f"expected sustained recovery at t=39 (not the t=30 transient), got {m['t_r_first_to_final_min']}"
    assert abs(m["TTR_first_to_final_min"] - 29.0) < 1e-6
    assert abs(m["TST_min"] - 19.0) < 1e-6
    assert abs(m["TTR_first_to_final_min"] - m["TST_min"] - 10.0) < 1e-6, "TTR - TST must equal the event's own duration - not interchangeable"
    expected_norm = ((10.0/nominal_th) - (2.0/nominal_th)) / ((39.0-10.0)/60.0)
    assert abs(m["recovery_rate_norm_per_h_first_to_final"] - expected_norm) < 1e-6
    print(f"  [A] TTR(first-to-final)={m['TTR_first_to_final_min']:.2f} min, TST={m['TST_min']:.2f} min, "
          f"R_rate={m['recovery_rate_norm_per_h_first_to_final']:.4f} h^-1 - match hand-calculated values")

def test_B_tut_noncontiguous():
    """Check: TUT integrates ALL below-threshold intervals, need not be
    continuous."""
    nominal_th = 10.0
    events = [{"event_type": "X", "start_min": 5.0, "end_min": 8.0}]
    values = [10.0]*5 + [5.0]*3 + [10.0]*5 + [6.0]*2 + [10.0]*5
    profile = _profile(0, 20, values)
    original_shift_hours = config.SHIFT_HOURS
    config.SHIFT_HOURS = 20.0 / 60.0
    try:
        m = calculate(profile, events, nominal_th)
    finally:
        config.SHIFT_HOURS = original_shift_hours
    assert abs(m["TUT_min"] - 5.0) < 1e-6
    assert abs(m["TUT_min"] - (14.0-5.0)) > 1e-6
    print(f"  [B] TUT={m['TUT_min']:.2f} min (two disjoint intervals summed correctly)")

def test_C_censored():
    """Check 2: one disruption without recovery before shift end - TTR is
    NOT 0, NOT assumed at shift end, and recovery rate is NaN."""
    nominal_th = 10.0
    events = [{"event_type": "X", "start_min": 10.0, "end_min": 20.0}]
    values = [10.0]*10 + [5.0]*21
    profile = _profile(0, 31, values)
    original_shift_hours = config.SHIFT_HOURS
    config.SHIFT_HOURS = 30.0 / 60.0
    try:
        m = calculate(profile, events, nominal_th)
    finally:
        config.SHIFT_HOURS = original_shift_hours
    assert m["recovered_within_horizon"] is False
    assert m["TTR_first_to_final_censored"] is True
    assert math.isnan(m["TTR_first_to_final_min"])
    assert abs(m["TTR_first_to_final_followup_min"] - 20.0) < 1e-6
    assert math.isnan(m["recovery_rate_norm_per_h_first_to_final"])
    assert math.isnan(m["TTR_event_mean_min"])
    assert m["total_recovery_interval_censored"] is True
    print(f"  [C] TTR correctly censored (NaN), follow-up={m['TTR_first_to_final_followup_min']:.1f} min, "
          f"total_recovery_interval={m['total_recovery_interval_min']:.1f} min (censored)")

def test_E_multi_event_ttr_definitions():
    """Check 3: multiple disruptions with complete recovery between events -
    total_recovery_interval must exclude the normal-operation gap, unlike
    TTR_first_to_final."""
    nominal_th = 10.0
    events = [{"event_type": "A", "start_min": 10.0, "end_min": 15.0},
              {"event_type": "B", "start_min": 40.0, "end_min": 45.0}]
    values = []
    values += [10.0]*10
    values += [2.0]*5
    values += [2.0 + (t-15)*2.0 for t in range(15, 20)]
    values += [10.0]*20
    values += [2.0]*5
    values += [2.0 + (t-45)*2.0 for t in range(45, 50)]
    values += [10.0]*31

    profile = _profile(0, 81, values)
    original_shift_hours = config.SHIFT_HOURS
    config.SHIFT_HOURS = 80.0 / 60.0
    try:
        m = calculate(profile, events, nominal_th)
    finally:
        config.SHIFT_HOURS = original_shift_hours

    assert m["event_TTR_A_identifiable"] is True and abs(m["event_TTR_A_min"] - 9.0) < 1e-6
    assert m["event_TTR_B_identifiable"] is True and abs(m["event_TTR_B_min"] - 9.0) < 1e-6
    assert m["event_TTR_A_status"] == "recovered_before_next_disruption"
    assert m["event_TTR_A_part_of_compound_episode"] is False
    assert abs(m["TTR_first_to_final_min"] - 39.0) < 1e-6
    assert abs(m["total_recovery_interval_min"] - 18.0) < 1e-6, f"expected 9+9=18 (gap excluded), got {m['total_recovery_interval_min']}"
    assert abs(m["TTR_event_mean_min"] - 9.0) < 1e-6
    assert m["total_recovery_interval_min"] < m["TTR_first_to_final_min"]
    assert m["total_recovery_interval_censored"] is False
    print(f"  [E] TTR_first_to_final={m['TTR_first_to_final_min']:.1f} min (spans the 20-min recovered gap), "
          f"total_recovery_interval={m['total_recovery_interval_min']:.1f} min (excludes it), TTR_event_mean={m['TTR_event_mean_min']:.1f} min")

def test_G_overlapping_disruptions():
    """Check 4: overlapping disruptions without intermediate recovery must
    be merged into one compound episode - no individual TTR is invented for
    either constituent event."""
    nominal_th = 10.0
    # B starts (15) before A's own end (20): literal overlap.
    events = [{"event_type": "A", "start_min": 10.0, "end_min": 20.0},
              {"event_type": "B", "start_min": 15.0, "end_min": 25.0}]
    values = []
    values += [10.0]*10                                   # t=0..9
    values += [2.0]*15                                    # t=10..24: combined trough (both disruptions)
    values += [2.0 + (t-25)*2.0 for t in range(25, 30)]    # t=25..29: ramp to 10
    values += [10.0]*11                                    # t=30..40

    profile = _profile(0, 41, values)
    original_shift_hours = config.SHIFT_HOURS
    config.SHIFT_HOURS = 40.0 / 60.0
    try:
        m = calculate(profile, events, nominal_th)
    finally:
        config.SHIFT_HOURS = original_shift_hours

    assert m["event_TTR_A_part_of_compound_episode"] is True
    assert m["event_TTR_B_part_of_compound_episode"] is True
    assert math.isnan(m["event_TTR_A_min"]), "no individual TTR must be invented for a compound-episode constituent"
    assert math.isnan(m["event_TTR_B_min"])
    assert m["event_TTR_A_status"] == "overlapping_superseded"
    assert m["event_TTR_B_status"] == "overlapping_superseded"
    assert abs(m["event_TTR_A_compound_TTR_min"] - 19.0) < 1e-6, f"combined episode TTR should be 29-10=19, got {m['event_TTR_A_compound_TTR_min']}"
    assert m["event_TTR_A_compound_TTR_min"] == m["event_TTR_B_compound_TTR_min"], "both constituents must point to the SAME compound episode TTR"
    assert abs(m["total_recovery_interval_min"] - 19.0) < 1e-6, "one merged episode, not double-counted"
    assert m["total_recovery_interval_n_episodes"] == 1
    print(f"  [G] Overlapping A/B merged into one compound episode, TTR={m['event_TTR_A_compound_TTR_min']:.1f} min, "
          f"total_recovery_interval={m['total_recovery_interval_min']:.1f} min (not double-counted)")

def test_J_nonoverlapping_schedule_no_intermediate_recovery():
    """Episodes must be identified from ACTUAL
    sustained throughput recovery, not merely from whether the scheduled
    disruption windows overlap. A and B's schedules do NOT overlap
    (A ends at t=15, B starts at t=50) but the trajectory never achieves
    sustained recovery between them - they must still be merged into one
    compound episode, not reported as two independent ones (which would
    misattribute B's TTR to only 10 min instead of the true 50)."""
    nominal_th = 10.0
    events = [{"event_type": "A", "start_min": 10.0, "end_min": 15.0},
              {"event_type": "B", "start_min": 50.0, "end_min": 55.0}]
    values = []
    values += [10.0]*10   # t=0..9
    values += [5.0]*40    # t=10..49: degraded continuously from A, never recovers before B - schedule gap, no actual recovery
    values += [2.0]*10    # t=50..59: B's own trough
    values += [10.0]*16   # t=60..75: instant sustained recovery at t=60

    profile = _profile(0, 76, values)
    original_shift_hours = config.SHIFT_HOURS
    config.SHIFT_HOURS = 75.0 / 60.0
    try:
        m = calculate(profile, events, nominal_th)
    finally:
        config.SHIFT_HOURS = original_shift_hours

    assert m["event_TTR_A_part_of_compound_episode"] is True
    assert m["event_TTR_B_part_of_compound_episode"] is True
    assert m["event_TTR_A_status"] == "overlapping_superseded"
    assert math.isnan(m["event_TTR_A_min"]) and math.isnan(m["event_TTR_B_min"])
    assert abs(m["event_TTR_A_compound_TTR_min"] - 50.0) < 1e-6, f"true compound TTR is 60-10=50 (degradation started at A, not B), got {m['event_TTR_A_compound_TTR_min']}"
    assert m["total_recovery_interval_n_episodes"] == 1, "A and B must be ONE episode despite non-overlapping schedules"
    assert abs(m["total_recovery_interval_min"] - 50.0) < 1e-6
    print(f"  [J] Non-overlapping schedules (A ends t=15, B starts t=50), no intermediate recovery -> "
          f"correctly merged into ONE compound episode, TTR={m['event_TTR_A_compound_TTR_min']:.1f} min "
          f"(not misattributed as B-only TTR=10 min)")

def test_H_recovery_at_horizon():
    """Check 6: recovery achieved exactly at the simulation horizon must
    not crash and must be correctly identified as recovered."""
    nominal_th = 10.0
    events = [{"event_type": "X", "start_min": 10.0, "end_min": 15.0}]
    values = [10.0]*10 + [2.0]*5 + [2.0 + (t-15)*2.0 for t in range(15, 20)]  # t=19 (last index) reaches 10.0
    profile = _profile(0, 20, values)
    original_shift_hours = config.SHIFT_HOURS
    config.SHIFT_HOURS = 19.0 / 60.0
    try:
        m = calculate(profile, events, nominal_th)
    finally:
        config.SHIFT_HOURS = original_shift_hours
    assert m["recovered_within_horizon"] is True
    assert abs(m["t_r_first_to_final_min"] - 19.0) < 1e-6
    assert abs(m["TTR_first_to_final_min"] - 9.0) < 1e-6
    print(f"  [H] Recovery exactly at horizon (t=19) correctly identified, TTR={m['TTR_first_to_final_min']:.1f} min, no crash")

def test_I_no_degradation():
    """Check 7: a scheduled event with no actual degradation must not
    crash, must report TTR=0 (recovery immediate, genuinely at t_e), and an
    undefined (NaN) recovery rate (check 8: zero-duration t_d->t_r
    interval), not a divide-by-zero or fabricated rate."""
    nominal_th = 10.0
    events = [{"event_type": "X", "start_min": 10.0, "end_min": 20.0}]
    values = [10.0] * 31
    profile = _profile(0, 31, values)
    original_shift_hours = config.SHIFT_HOURS
    config.SHIFT_HOURS = 30.0 / 60.0
    try:
        m = calculate(profile, events, nominal_th)
    finally:
        config.SHIFT_HOURS = original_shift_hours
    assert m["recovered_within_horizon"] is True
    assert abs(m["TTR_first_to_final_min"] - 0.0) < 1e-6
    assert abs(m["PL_cartons"] - 0.0) < 1e-6
    assert abs(m["TUT_min"] - 0.0) < 1e-6
    assert math.isnan(m["recovery_rate_norm_per_h_first_to_final"]), "zero-duration t_d->t_r interval must give an undefined (NaN) rate, not 0 or inf"
    print(f"  [I] No-degradation event: TTR={m['TTR_first_to_final_min']:.1f} min, recovery_rate=NaN (undefined interval), no crash")

def test_K_different_nominal_throughput():
    """Check 9: recovery rate must use DESIGN-SPECIFIC nominal throughput -
    two designs with the same RELATIVE degradation/recovery shape but
    different absolute nominal throughput must give the SAME normalized
    rate (h^-1) but a proportionally different absolute (TH/h^2) rate."""
    nominal_1, nominal_2 = 10.0, 20.0  # design 2 = 2x design 1
    events = [{"event_type": "X", "start_min": 10.0, "end_min": 15.0}]

    def make_m(nominal):
        # same RELATIVE shape for both: trough at 20% of nominal, instant
        # full recovery to 100% of nominal at t=20.
        values = [nominal]*10 + [0.2*nominal]*10 + [nominal]*11  # t=0..30
        profile = _profile(0, 31, values)
        original_shift_hours = config.SHIFT_HOURS
        config.SHIFT_HOURS = 30.0 / 60.0
        try:
            return calculate(profile, events, nominal)
        finally:
            config.SHIFT_HOURS = original_shift_hours

    m1, m2 = make_m(nominal_1), make_m(nominal_2)
    assert abs(m1["t_d_first_to_final_min"] - m2["t_d_first_to_final_min"]) < 1e-9
    assert abs(m1["t_r_first_to_final_min"] - m2["t_r_first_to_final_min"]) < 1e-9, "same relative shape -> same recovery timing regardless of nominal_th"
    assert abs(m1["recovery_rate_norm_per_h_first_to_final"] - m2["recovery_rate_norm_per_h_first_to_final"]) < 1e-9, \
        "normalized recovery rate must be invariant to nominal_th for proportionally-scaled trajectories"
    assert abs(m2["recovery_rate_TH_per_h2_first_to_final"] - 2.0 * m1["recovery_rate_TH_per_h2_first_to_final"]) < 1e-9, \
        "the NON-normalized (TH/h^2) rate must scale WITH nominal_th (2x nominal -> 2x absolute rate), unlike the normalized one"
    print(f"  [K] normalized rate identical across nominal_th={nominal_1:.0f} and {nominal_2:.0f} "
          f"({m1['recovery_rate_norm_per_h_first_to_final']:.4f} h^-1 both); "
          f"absolute rate scales 2x ({m1['recovery_rate_TH_per_h2_first_to_final']:.2f} -> {m2['recovery_rate_TH_per_h2_first_to_final']:.2f})")

def test_L_aggregation_consistency():
    """Check 10: the episode-weighted ("pooled") and realization-weighted
    scenario-level R_rate aggregates (analysis.py) must each be internally
    consistent with their own definition, and must NOT be silently
    interchangeable - they differ exactly when realizations contribute
    unequal numbers of identifiable/rate-defined episodes, which is the
    case this test is designed to catch."""
    # 3 synthetic realizations' own identifiable+rate-defined episode rates.
    per_realization_rates = [[1.0, 3.0], [2.0], [4.0]]  # realization 1 has 2 episodes, 2 and 3 have 1 each

    pooled = [v for r in per_realization_rates for v in r]
    pooled_mean = sum(pooled) / len(pooled)  # (1+3+2+4)/4 = 2.5 - episode-weighted

    realization_means = [sum(r) / len(r) for r in per_realization_rates]  # [2.0, 2.0, 4.0]
    realization_mean = sum(realization_means) / len(realization_means)  # (2+2+4)/3 = 2.6667 - realization-weighted

    assert abs(pooled_mean - 2.5) < 1e-9
    assert abs(realization_mean - 8.0/3.0) < 1e-9
    assert abs(pooled_mean - realization_mean) > 1e-6, \
        "pooled (episode-weighted) and realization-weighted means must differ when episode counts vary by realization - analysis.py reports them as two distinct, non-interchangeable aggregates rather than silently mixing them"
    print(f"  [L] episode-weighted pooled_mean={pooled_mean:.4f} != realization-weighted mean={realization_mean:.4f} "
          f"- the two scenario-level R_rate aggregates are distinct")

def test_F_objective_direction_and_dominance():
    """Check 10: R_phi is the only resilience objective, F = [-TH, E, -R_phi]; Pareto dominance on these
    minimisation-convention objectives and the representative selection (including "max_resilience") are consistent."""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    from .optimisation import _resilience_f3, representatives, verify_nondominated_bruteforce

    assert not hasattr(config, "RESILIENCE_OBJECTIVE"), "no selectable resilience objective: R_phi is the only one"
    assert not hasattr(config, "TTR_SCENARIO_DEFINITION"), "TTR is not an optimisation objective"

    # B dominates A and C in every objective (higher TH, lower E, higher R_phi).
    A = {"disrupted_throughput_mean": 2.0, "energy_kWh_mean": 20.0, "R_phi_tf_mean": 0.90}
    B = {"disrupted_throughput_mean": 2.5, "energy_kWh_mean": 18.0, "R_phi_tf_mean": 0.95}
    C = {"disrupted_throughput_mean": 1.5, "energy_kWh_mean": 25.0, "R_phi_tf_mean": 0.80}
    candidates = [A, B, C]

    F = np.asarray([[-c["disrupted_throughput_mean"], c["energy_kWh_mean"], _resilience_f3(c)] for c in candidates])
    assert np.allclose(F[:, 2], [-0.90, -0.95, -0.80]), "objective vector must be F=[-TH, E, -R_phi]"

    pymoo_mask = np.zeros(3, dtype=bool)
    pymoo_mask[NonDominatedSorting().do(F, only_non_dominated_front=True)] = True
    bruteforce_mask = verify_nondominated_bruteforce(F)
    assert np.array_equal(pymoo_mask, bruteforce_mask)
    assert list(pymoo_mask) == [False, True, False], f"expected only B non-dominated, got {pymoo_mask}"

    reps = representatives(candidates)
    assert reps["max_throughput"] == B and reps["min_energy"] == B and reps["max_resilience"] == B and reps["middle_balanced"] == B
    print("  [F] F=[-TH,E,-R_phi]; pymoo and pairwise dominance agree; representatives consistent")

def run():
    print("Running resilience-metric validation tests...")
    test_A_ttr_tst_transient_recoveryrate()
    test_B_tut_noncontiguous()
    test_C_censored()
    test_E_multi_event_ttr_definitions()
    test_G_overlapping_disruptions()
    test_J_nonoverlapping_schedule_no_intermediate_recovery()
    test_H_recovery_at_horizon()
    test_I_no_degradation()
    test_K_different_nominal_throughput()
    test_L_aggregation_consistency()
    test_F_objective_direction_and_dominance()
    print("All validation tests PASSED.")

if __name__ == "__main__":
    run()
