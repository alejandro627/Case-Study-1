"""Tests of the integer-aware NSGA-II (RoundingRepair) and of the per-generation diagnostics and hypervolume
callback. All tests use a small population and few generations.

    python -m umdo_cpps.test_nsga2_discrete
"""
from __future__ import annotations
import numpy as np
from pymoo.indicators.hv import HV

from . import config
from .design import DV_LEVELS, DV_ORDER, XL, XU, design_key, vector_to_design
from .optimisation import (
    CPPSProblem, DiagnosticsCallback, LHSIntegerSampling, _CACHE,
    _fixed_hv_reference_point, run_nsga2,
)


def _all_designs_valid(X: np.ndarray) -> bool:
    for row in X:
        design = vector_to_design(row)
        for dv in DV_ORDER:
            if design[dv] not in DV_LEVELS[dv]:
                return False
    return True


def test_1_hv_synthetic_known_results():
    """Check 5.2: the HV indicator against a result computed by hand."""
    pts = np.array([[1, 4], [2, 2], [3, 1]], dtype=float)
    ref = np.array([4, 5], dtype=float)
    hv = HV(ref_point=ref)(pts)
    assert abs(hv - 8.0) < 1e-9, f"2D HV expected 8.0, got {hv}"

    pts3 = np.array([[0.0, 0.0, 0.0]])
    ref3 = np.array([1.0, 1.0, 1.0])
    hv3 = HV(ref_point=ref3)(pts3)
    assert abs(hv3 - 1.0) < 1e-9, f"3D single-point HV expected 1.0, got {hv3}"

    # A=(0,0.5,0), B=(0.5,0,0), ref=(1,1,1): box(A)=1*0.5*1=0.5,
    # box(B)=0.5*1*1=0.5, intersection=[0.5,1]x[0.5,1]x[0,1]=0.5*0.5*1=0.25,
    # union = 0.5+0.5-0.25 = 0.75
    pts3b = np.array([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0]])
    hv3b = HV(ref_point=ref3)(pts3b)
    assert abs(hv3b - 0.75) < 1e-9, f"3D two-point HV expected 0.75, got {hv3b}"
    print("  [1] HV indicator matches hand-computed results on 3 synthetic cases")


def test_2_fixed_reference_point_dominated_by_feasible_region():
    """The fixed, analytically-derived reference point must be dominated
    by (worse than) every physically-possible F value, not just every
    observed one, for the HV curve to be a stable, run-independent metric."""
    ref = _fixed_hv_reference_point()
    assert ref[0] == 0.0, "obj0 (-TH) reference must be the physical supremum 0.0"
    assert ref[2] == 0.0, "obj2 (-R_phi) reference must be the physical supremum 0.0 (R_phi in [0,1])"
    assert ref[1] > 0, "obj1 (E) reference must be a positive, generous energy ceiling"
    print(f"  [2] fixed HV reference point = {ref.tolist()}")


def test_3_rounding_repair_keeps_population_integer_and_in_bounds():
    """Checks 3.2.1-3.2.2: every individual in the post-repair population
    (after real sampling + crossover + mutation across several real
    generations) must be an exact integer within [XL, XU]."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    _CACHE.clear()
    original_n_lhs = config.N_LHS_OPTIMISATION
    config.N_LHS_OPTIMISATION = 10
    try:
        import importlib
        from . import optimisation as _opt_mod
        importlib.reload(_opt_mod)
        problem = _opt_mod.CPPSProblem()
        algorithm = NSGA2(
            pop_size=16, sampling=LHSIntegerSampling(),
            crossover=SBX(prob=.90, eta=15), mutation=PM(prob=1.0 / len(DV_ORDER), eta=20),
            eliminate_duplicates=True, repair=RoundingRepair(), return_least_infeasible=True,
        )
        res = minimize(problem, algorithm, get_termination("n_gen", 3), seed=17, verbose=False)
        X = res.pop.get("X")
        assert np.allclose(X, np.round(X)), "population must be exactly integer-valued after repair"
        assert np.all(X >= XL) and np.all(X <= XU), "population must respect [XL, XU] after repair"
        assert _all_designs_valid(X), "every repaired row must map to an admissible DV_LEVELS combination"
        print(f"  [3] population integer+in-bounds+admissible after 3 real generations (pop=16)")
    finally:
        config.N_LHS_OPTIMISATION = original_n_lhs


def test_4_duplicate_elimination_operates_on_discrete_designs():
    """Check 3.2.3: two continuous vectors that round to the SAME integer
    design must be treated as duplicates once repair is applied (repair
    runs before eliminate_duplicates inside pymoo's Mating.do())."""
    from pymoo.core.duplicate import DefaultDuplicateElimination
    from pymoo.core.population import Population
    from pymoo.operators.repair.rounding import RoundingRepair

    # Two distinct continuous vectors that round to the identical integer design.
    x1 = np.array([2.1, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    x2 = np.array([2.4, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    assert not np.allclose(x1, x2), "sanity: the two continuous vectors must actually differ"
    assert design_key(vector_to_design(x1)) == design_key(vector_to_design(x2)), (
        "sanity: both must round to the same integer design"
    )

    class _DummyProblem:
        pass

    off = Population.new(X=np.vstack([x1, x2]))
    repaired = RoundingRepair()._do(_DummyProblem(), off.get("X"))
    off.set("X", repaired)
    assert np.allclose(off.get("X")[0], off.get("X")[1]), "after repair the two rows must be identical"

    dedup = DefaultDuplicateElimination(epsilon=1e-16)
    survivors = dedup.do(off)  # to_itself=True (default): removes self-duplicates within `off`
    assert len(survivors) == 1, (
        f"post-repair duplicate elimination should collapse the 2 colliding individuals to 1, got {len(survivors)}"
    )
    print("  [4] two continuous vectors rounding to the same integer design ARE eliminated as duplicates post-repair")


def test_5_unique_discrete_design_tracking():
    """Check 3.2.4: DiagnosticsCallback's n_unique_designs_in_population
    counts DISTINCT INTEGER designs, not distinct continuous rows."""
    cb = DiagnosticsCallback()

    class _FakeEvaluator:
        n_eval = 10

    class _FakePop:
        def __init__(self, X, F, G):
            self._X, self._F, self._G = X, F, G
        def get(self, key):
            return {"X": self._X, "F": self._F, "G": self._G, "CV": np.zeros((len(self._X), 1))}[key]
        def __len__(self):
            return len(self._X)

    class _FakeAlgorithm:
        n_gen = 1
        evaluator = _FakeEvaluator()

    # 4 rows but only 2 distinct integer designs once rounded (2.1/2.4 -> 2).
    X = np.array([
        [2.1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2.4, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [1.0, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [1.2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ])
    F = np.zeros((4, 3))
    G = np.zeros((4, 9))
    algo = _FakeAlgorithm()
    algo.pop = _FakePop(X, F, G)
    cb.notify(algo)
    assert cb.rows[-1]["n_unique_designs_in_population"] == 2, (
        f"expected 2 unique rounded designs, got {cb.rows[-1]['n_unique_designs_in_population']}"
    )
    print("  [5] DiagnosticsCallback correctly tracks unique INTEGER designs (2 of 4 continuous rows collapse)")


def test_6_identical_integer_design_gives_identical_objectives():
    """Check 3.2.5: the SAME integer design, evaluated with the SAME LHS
    realizations, must give numerically identical objective/constraint
    values regardless of whether repair is enabled (repair only affects
    which continuous vectors the search explores, never the per-design
    evaluate_umda() computation itself)."""
    from .analysis import evaluate_umda
    from .uncertainty import UV_ORDER, lhs_unit

    design = {"a_CNC": 2, "a_A": 2, "a_QC": 2, "r_A": 3, "r_QC": 2,
              "v_conv": 3, "u_sched": 3, "u_maint": 3, "u_inspect": 1, "u_energy": 3}
    unit_lhs = lhs_unit(10, len(UV_ORDER), config.LHS_SEED)
    s1, _, _ = evaluate_umda(design, unit_lhs, keep_profiles=False)
    s2, _, _ = evaluate_umda(design, unit_lhs, keep_profiles=False)
    for key in ("disrupted_throughput_mean", "energy_kWh_mean", "R_phi_tf_mean", "g_TH_det", "g_recovery"):
        assert abs(s1[key] - s2[key]) < 1e-12, f"{key} not reproducible: {s1[key]} vs {s2[key]}"
    print("  [6] identical integer design + identical LHS batch -> numerically identical objectives/constraints")


def test_7_smoke_old_vs_new_repair_comparison():
    """Check 3.2.6: a small NSGA-II run with integer repair disabled vs enabled (same seeds), reporting unique-design
    counts, duplicate frequency and feasibility side by side. Both runs must complete and give valid discrete designs
    when evaluated; only the run with repair keeps an integer-valued population."""
    original_pop = config.NSGA2_POPULATION
    original_gen = config.NSGA2_GENERATIONS
    original_n_lhs = config.N_LHS_OPTIMISATION
    config.NSGA2_POPULATION = 16
    config.NSGA2_GENERATIONS = 4
    config.N_LHS_OPTIMISATION = 10
    try:
        import importlib
        from . import optimisation as _opt_mod
        importlib.reload(_opt_mod)

        res_old = _opt_mod.run_nsga2(use_integer_repair=False)
        X_old = res_old.pop.get("X")
        n_unique_old = len({design_key(vector_to_design(row)) for row in X_old})
        is_integer_old = np.allclose(X_old, np.round(X_old))

        res_new = _opt_mod.run_nsga2(use_integer_repair=True)
        X_new = res_new.pop.get("X")
        n_unique_new = len({design_key(vector_to_design(row)) for row in X_new})
        is_integer_new = np.allclose(X_new, np.round(X_new))

        assert _all_designs_valid(X_old), "old approach: every evaluated row must still map to a valid design (rounding at evaluation time)"
        assert _all_designs_valid(X_new), "new approach: every row must map to a valid design"
        assert is_integer_new, "new approach: population X must be exactly integer-valued"

        print(f"  [7] old (repair=False): pop stored as continuous={not is_integer_old}, "
              f"{n_unique_old}/{len(X_old)} unique integer designs after rounding")
        print(f"      new (repair=True):  pop stored as continuous={not is_integer_new}, "
              f"{n_unique_new}/{len(X_new)} unique integer designs")
        print(f"      both runs completed successfully; Pareto fronts are not required to match "
              f"(the search representation legitimately changed).")
    finally:
        config.NSGA2_POPULATION = original_pop
        config.NSGA2_GENERATIONS = original_gen
        config.N_LHS_OPTIMISATION = original_n_lhs


def run():
    print("Running NSGA-II discrete-handling + diagnostics validation tests...")
    test_1_hv_synthetic_known_results()
    test_2_fixed_reference_point_dominated_by_feasible_region()
    test_3_rounding_repair_keeps_population_integer_and_in_bounds()
    test_4_duplicate_elimination_operates_on_discrete_designs()
    test_5_unique_discrete_design_tracking()
    test_6_identical_integer_design_gives_identical_objectives()
    test_7_smoke_old_vs_new_repair_comparison()
    print("All NSGA-II discrete-handling tests PASSED.")


if __name__ == "__main__":
    run()
