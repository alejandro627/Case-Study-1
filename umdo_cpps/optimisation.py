from __future__ import annotations
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.termination import get_termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
from . import config
from .analysis import evaluate_umda
from .deterministic_adapter import thresholds as deterministic_thresholds
from .design import DV_ORDER, XL, XU, design_key, vector_to_design
from .uncertainty import UV_ORDER, lhs_unit

_CACHE: Dict[tuple, Dict[str, Any] | None] = {}
_OPT_LHS = lhs_unit(config.N_LHS_OPTIMISATION, len(UV_ORDER), config.LHS_SEED)

# Non-finite F or G values are replaced by this large finite value before they are passed to pymoo: NaN comparisons
# can make an individual look feasible in the tournament, and inf/NaN become empty cells in the Excel export. Under
# the minimisation convention the value is the worst possible for every objective and constraint.
_LARGE = 1e6

_G_KEYS = [
    "g_TH_det", "g_WIP_det", "g_CT_det", "g_E_det", "g_e_carton_det", "g_MDA",
    "g_TH_prob", "g_E_prob", "g_recovery",
]

def _sanitize(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.where(np.isfinite(arr), arr, _LARGE)

def _resilience_f3(r: Mapping[str, Any]) -> float:
    """Third UMDO objective in the pymoo minimisation convention: maximise R_phi(t_f) -> minimise -R_phi(t_f).
        R_phi(t_f) is the only resilience objective; TTR, recovery rate, PL, TST and TUT are reported only. The single
        function keeps the formula identical in _evaluate, refine_pareto_front and representatives.
    """
    return -float(r["R_phi_tf_mean"])

def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_default))
        f.write("\n")

class CPPSProblem(ElementwiseProblem):
    """Evaluation counters (n_success, n_failed_exceptions, n_cache_hits) of this problem instance, i.e. of one
        run_nsga2() call.
    """
    def __init__(self, log_path: Optional[Path] = None, fail_fast_threshold: Optional[int] = None):
        super().__init__(n_var=len(DV_ORDER), n_obj=3, n_ieq_constr=len(_G_KEYS), xl=XL, xu=XU, vtype=int)
        self.log_path = log_path
        self.fail_fast_threshold = (
            config.NSGA2_FAIL_FAST_THRESHOLD if fail_fast_threshold is None else fail_fast_threshold
        )
        self.n_success = 0
        self.n_failed_exceptions = 0
        self.n_legitimately_infeasible = 0
        self.n_cache_hits = 0

    def _log_exception(self, design: Dict[str, int], exc: Exception) -> None:
        record = {
            "timestamp": datetime.now().isoformat(),
            "design": design,
            "exception_type": type(exc).__name__,
            "exception_repr": repr(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"[CPPSProblem] UNEXPECTED evaluation exception for design={design}: {exc!r} "
              f"(failure #{self.n_failed_exceptions} this run)")
        if self.log_path is not None:
            _append_jsonl(self.log_path, record)

    def _evaluate(self, x, out, *args, **kwargs):
        design=vector_to_design(x)
        key=design_key(design)
        if key not in _CACHE:
            try:
                r = evaluate_umda(design,_OPT_LHS,keep_profiles=False)[0]
                _CACHE[key] = r
                self.n_success += 1
                if not r["feasible_overall"]:
                    self.n_legitimately_infeasible += 1
            except Exception as exc:
                self.n_failed_exceptions += 1
                self._log_exception(design, exc)
                _CACHE[key]=None
                if self.n_failed_exceptions >= self.fail_fast_threshold:
                    raise RuntimeError(
                        f"NSGA-II fail-fast: {self.n_failed_exceptions} distinct designs raised an "
                        f"UNEXPECTED exception during evaluation (threshold={self.fail_fast_threshold}). "
                        f"This is NOT ordinary constraint infeasibility - it indicates an implementation "
                        f"bug, not a hard design. See {self.log_path or '(no log_path configured)'} for "
                        f"full tracebacks. Last error: {exc!r}"
                    ) from exc
        else:
            self.n_cache_hits += 1
        r=_CACHE[key]
        if r is None:
            out["F"]=np.full(3,_LARGE)
            out["G"]=np.full(len(_G_KEYS),_LARGE)
            return
        out["F"]=_sanitize([
            -r["disrupted_throughput_mean"],
            r["energy_kWh_mean"],
            _resilience_f3(r),
        ])
        out["G"]=_sanitize([r[k] for k in _G_KEYS])

    def evaluation_summary(self) -> Dict[str, Any]:
        """Execution summary: constraint infeasibility (a normal outcome) is reported separately from unexpected evaluation
                exceptions (implementation errors).
        """
        n_distinct = self.n_success + self.n_failed_exceptions
        return {
            "n_distinct_designs_evaluated": n_distinct,
            "n_evaluated_successfully": self.n_success,
            "n_legitimately_infeasible": self.n_legitimately_infeasible,
            "n_feasible": self.n_success - self.n_legitimately_infeasible,
            "n_failed_with_exception": self.n_failed_exceptions,
            "n_cache_hits_this_run": self.n_cache_hits,
            "cache_hit_rate": self.n_cache_hits / (n_distinct + self.n_cache_hits) if (n_distinct + self.n_cache_hits) else 0.0,
        }

class LHSIntegerSampling(Sampling):
    """Latin hypercube initial population over the 10 integer design variables (replaces pymoo's
        IntegerRandomSampling). Uses lhs_unit() from uncertainty.py, maps the unit cube onto [XL, XU] and rounds to the
        nearest level (every DV has a contiguous integer range). Seeded with config.POPULATION_LHS_SEED.
    """
    def _do(self, problem, n_samples, **kwargs):
        unit = lhs_unit(n_samples, problem.n_var, config.POPULATION_LHS_SEED)
        x = problem.xl + unit * (problem.xu - problem.xl)
        return np.round(x).astype(int)

def _fixed_hv_reference_point() -> np.ndarray:
    """Fixed hypervolume reference point in F-space [-TH_mean, E_mean, -R_phi_mean] (minimisation convention),
        independent of the data so that runs can be compared:
          - -TH: throughput >= 0, so 0.0 is an upper bound;
          - E: 1.5 x E_TOT_MAX (deterministic_adapter.thresholds()), above the energy of every feasible design;
          - -R_phi: R_phi(t_f) is clamped to [0, 1], so 0.0 is an upper bound.
    """
    th = deterministic_thresholds()
    return np.array([0.0, 1.5 * th["E_TOT_MAX"], 0.0])

class DiagnosticsCallback(Callback):
    """Per-generation convergence diagnostics computed during the run through pymoo's Callback.notify() (no
        save_history, so memory use does not grow with the number of generations). Every row has the same keys; the
        '_best_feasible' fields are NaN when no individual is feasible.
    """
    def __init__(self, csv_path: Optional[Path] = None,
                 checkpoint_path: Optional[Path] = None,
                 checkpoint_every_n_gen: Optional[int] = None,
                 ref_point: Optional[np.ndarray] = None):
        super().__init__()
        self.csv_path = csv_path
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every_n_gen = checkpoint_every_n_gen or config.NSGA2_CHECKPOINT_EVERY_N_GEN
        self.ref_point = ref_point if ref_point is not None else _fixed_hv_reference_point()
        self._hv = HV(ref_point=self.ref_point)
        self.rows: List[Dict[str, Any]] = []
        self._wrote_header = False

    def notify(self, algorithm):
        pop = algorithm.pop
        X = pop.get("X")
        F = pop.get("F")
        G = pop.get("G")
        CV_raw = pop.get("CV")
        CV = CV_raw.reshape(-1) if CV_raw is not None else np.zeros(len(pop))

        feasible_mask = CV <= 0
        n_feasible = int(np.sum(feasible_mask))
        F_feas = F[feasible_mask] if n_feasible > 0 else np.empty((0, F.shape[1]))

        # Diversity of the current (repaired, integer) population.
        keys = {tuple(int(round(v)) for v in row) for row in X}
        n_unique_designs = len(keys)

        if n_feasible > 0:
            nd_idx = NonDominatedSorting().do(F_feas, only_non_dominated_front=True)
            F_nd = F_feas[nd_idx]
            n_nds_feasible = len(F_nd)
            # Points beyond the fixed reference point are clipped (pymoo's HV requires every point to dominate it);
            # the number of clipped points is recorded in hv_ref_point_points_clipped.
            n_clipped = int(np.sum(np.any(F_nd > self.ref_point, axis=1)))
            clipped = np.minimum(F_nd, self.ref_point - 1e-9)
            hv = float(self._hv(clipped))
        else:
            n_nds_feasible = 0
            hv = 0.0
            n_clipped = 0

        row: Dict[str, Any] = {
            "n_gen": int(algorithm.n_gen),
            "n_eval": int(algorithm.evaluator.n_eval),
            "pop_size": int(len(pop)),
            "n_unique_designs_in_population": n_unique_designs,
            "n_feasible_in_population": n_feasible,
            "cv_min": float(np.min(CV)) if len(CV) else float("nan"),
            "cv_mean": float(np.mean(CV)) if len(CV) else float("nan"),
            "n_nondominated_feasible": n_nds_feasible,
            "hypervolume": hv,
            "hv_ref_point_points_clipped": n_clipped,
        }
        for j, name in enumerate(("neg_throughput", "energy", "neg_R_phi_tf")):
            row[f"F{j}_{name}_best"] = float(np.min(F[:, j]))
            row[f"F{j}_{name}_mean"] = float(np.mean(F[:, j]))
            row[f"F{j}_{name}_std"] = float(np.std(F[:, j]))
            row[f"F{j}_{name}_best_feasible"] = float(np.min(F_feas[:, j])) if F_feas.shape[0] else float("nan")

        self.rows.append(row)
        if self.csv_path is not None:
            self._append_row(row)
        if self.checkpoint_path is not None and int(algorithm.n_gen) % self.checkpoint_every_n_gen == 0:
            self._write_checkpoint(algorithm, X, F, G, CV, feasible_mask)

    def _append_row(self, row: Dict[str, Any]) -> None:
        import csv as csv_module
        write_header = not self._wrote_header
        mode = "w" if write_header else "a"
        with self.csv_path.open(mode, newline="", encoding="utf-8") as f:
            w = csv_module.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        self._wrote_header = True

    def _write_checkpoint(self, algorithm, X, F, G, CV, feasible_mask) -> None:
        """Write the current population to a single checkpoint file every checkpoint_every_n_gen generations (overwritten
                each time). The file preserves intermediate results; it is not used to resume a search.
        """
        from .exporters import write_csv
        rows = []
        for i in range(len(X)):
            design = vector_to_design(X[i])
            r: Dict[str, Any] = {"n_gen": int(algorithm.n_gen), "individual_index": i, **design}
            r["F_neg_throughput"], r["F_energy"], r["F_neg_R_phi_tf"] = (float(v) for v in F[i])
            for k, gk in enumerate(_G_KEYS):
                r[gk] = float(G[i, k])
            r["feasible"] = bool(feasible_mask[i])
            rows.append(r)
        write_csv(self.checkpoint_path, rows)

def run_nsga2(save_history: bool = False, out_dir: Optional[Path] = None,
              use_integer_repair: Optional[bool] = None):
    """Run NSGA-II.

        use_integer_repair (default config.NSGA2_USE_INTEGER_REPAIR): RoundingRepair is applied after sampling and after
        crossover/mutation in every generation (pymoo applies repair before duplicate elimination), so the population
        stays integer-valued and duplicate elimination and crowding distance work on the discrete design space.

        out_dir (optional): writes the per-generation diagnostics (nsga2_generations.csv), the population checkpoint
        (nsga2_checkpoint_latest.csv) and the evaluation-exception log (nsga2_exceptions.jsonl).

        The design cache _CACHE is cleared at the start of every call, because it is keyed by the design only and not by
        the LHS sample set used for its evaluation.
    """
    _CACHE.clear()

    use_repair = config.NSGA2_USE_INTEGER_REPAIR if use_integer_repair is None else use_integer_repair
    exception_log_path = (out_dir / "nsga2_exceptions.jsonl") if out_dir is not None else None
    problem = CPPSProblem(log_path=exception_log_path)

    diagnostics_csv = (out_dir / "nsga2_generations.csv") if out_dir is not None else None
    checkpoint_path = (out_dir / "nsga2_checkpoint_latest.csv") if out_dir is not None else None
    callback = DiagnosticsCallback(csv_path=diagnostics_csv, checkpoint_path=checkpoint_path)

    # save_history=True stores the full algorithm state of every generation in res.history (used by
    # hypervolume_study.py); off by default.
    algorithm=NSGA2(
        pop_size=config.NSGA2_POPULATION,
        sampling=LHSIntegerSampling(),
        crossover=SBX(prob=config.NSGA2_CROSSOVER_PROB,eta=config.NSGA2_CROSSOVER_ETA),
        mutation=PM(prob=1.0/len(DV_ORDER),eta=config.NSGA2_MUTATION_ETA),
        eliminate_duplicates=True,
        repair=RoundingRepair() if use_repair else None,
        return_least_infeasible=True,
    )
    res = minimize(
        problem, algorithm,
        get_termination("n_gen",config.NSGA2_GENERATIONS),
        seed=config.NSGA2_SEED, verbose=True, save_history=save_history,
        callback=callback,
    )
    res.cpps_problem = problem
    res.diagnostics_callback = callback

    summary = problem.evaluation_summary()
    print(f"[run_nsga2] evaluation summary: {summary}")
    if out_dir is not None:
        from .run_safety import write_json
        write_json(out_dir / "nsga2_evaluation_summary.json", summary)
    return res

def hypervolume_history(res, ref_point: np.ndarray | None = None) -> Tuple[List[Dict[str,Any]], np.ndarray]:
    """Per-generation hypervolume of the non-dominated front of the population (res.history[i].opt); requires
        run_nsga2(save_history=True).

        ref_point defaults to the worst value of each objective over the whole run plus 10 % of its observed range
        (additive, because two objectives are negative). The same reference point is used for every generation.
    """
    if res.history is None:
        raise ValueError("hypervolume_history requires run_nsga2(save_history=True).")
    from pymoo.indicators.hv import HV

    all_F = np.vstack([algo.opt.get("F") for algo in res.history])
    if ref_point is None:
        worst = all_F.max(axis=0)
        margin = 0.1 * (all_F.max(axis=0) - all_F.min(axis=0))
        ref_point = worst + np.where(margin <= 1e-9, 1.0, margin)

    hv = HV(ref_point=ref_point)
    rows = []
    for algo in res.history:
        F = algo.opt.get("F")
        rows.append({
            "n_gen": algo.n_gen,
            "n_eval": int(algo.evaluator.n_eval),
            "n_nds": int(len(F)),
            "hypervolume": float(hv(F)),
        })
    return rows, ref_point

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if a dominates b (minimisation): a <= b in every objective and a < b in at least one."""
    return bool(np.all(a <= b) and np.any(a < b))

def verify_nondominated_bruteforce(F: np.ndarray) -> np.ndarray:
    """Non-dominated mask of F by pairwise comparison (O(n^2)), independent of pymoo's NonDominatedSorting; used as a
        cross-check.
    """
    n = len(F)
    is_nd = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and _dominates(F[j], F[i]):
                is_nd[i] = False
                break
    return is_nd

def refine_pareto_front(optimisation_X, final_lhs) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]], Dict[str,int]]:
    """Re-evaluate every NSGA-II candidate at the high-fidelity sample size N_LHS_FINAL, keep the candidates that
        satisfy all nine constraints and recompute non-dominance on them (objective values change slightly between the
        two sample sizes). Candidates are de-duplicated by design_key() before the evaluation.

        Returns (final_front, all_candidates_reevaluated, stage_counts) with stage_counts = {n_evaluated_raw, n_unique,
        n_feasible, n_nondominated, n_nondominated_bruteforce_check}.
    """
    raw = np.atleast_2d(optimisation_X)
    n_evaluated_raw = len(raw)

    all_candidates: List[Dict[str,Any]] = []
    seen_keys: set = set()
    for x in raw:
        design=vector_to_design(x)
        key=design_key(design)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        summary,_,_=evaluate_umda(design,final_lhs,keep_profiles=False)
        all_candidates.append({
            "solution_id":len(all_candidates)+1,
            "lhs_sample_size":int(final_lhs.shape[0]),
            "lhs_seed":config.LHS_SEED+1000,
            **summary,
        })
    n_unique = len(all_candidates)

    feasible=[c for c in all_candidates if c["feasible_overall"]]
    n_feasible = len(feasible)
    if not feasible:
        return [], all_candidates, {"n_evaluated_raw":n_evaluated_raw,"n_unique":n_unique,"n_feasible":0,"n_nondominated":0,"n_nondominated_bruteforce_check":0}

    F=np.asarray([[-c["disrupted_throughput_mean"],c["energy_kWh_mean"],_resilience_f3(c)] for c in feasible],dtype=float)
    nd_idx=NonDominatedSorting().do(F,only_non_dominated_front=True)
    final_front=[feasible[i] for i in nd_idx]

    # Cross-check with the pairwise implementation; a mismatch raises an error.
    bruteforce_mask = verify_nondominated_bruteforce(F)
    n_nd_bruteforce = int(np.sum(bruteforce_mask))
    if n_nd_bruteforce != len(final_front):
        print(f"[refine_pareto_front] WARNING: pymoo NonDominatedSorting found {len(final_front)} "
              f"non-dominated candidates but the independent brute-force check found {n_nd_bruteforce} - "
              "investigate before trusting this front.")

    stage_counts = {
        "n_evaluated_raw": n_evaluated_raw, "n_unique": n_unique, "n_feasible": n_feasible,
        "n_nondominated": len(final_front), "n_nondominated_bruteforce_check": n_nd_bruteforce,
    }
    return final_front, all_candidates, stage_counts

def representatives(rows):
    # max_resilience = maximum R_phi(t_f).
    rows=[dict(r) for r in rows]
    max_th=max(rows,key=lambda r:r["disrupted_throughput_mean"])
    min_e=min(rows,key=lambda r:r["energy_kWh_mean"])
    max_r=max(rows,key=lambda r:r["R_phi_tf_mean"])
    matrix=np.asarray([[-r["disrupted_throughput_mean"],r["energy_kWh_mean"],_resilience_f3(r)] for r in rows],float)
    mn=matrix.min(axis=0); mx=matrix.max(axis=0); span=np.where(mx-mn<=1e-12,1,mx-mn)
    dist=np.linalg.norm((matrix-mn)/span,axis=1)
    balanced=rows[int(np.argmin(dist))]
    return {"max_throughput":max_th,"min_energy":min_e,"max_resilience":max_r,"middle_balanced":balanced}

_DISPLAY_LABELS = {
    "max_throughput": "Maximum Throughput",
    "min_energy": "Minimum Energy",
    "max_resilience": "Maximum Resilience (R_phi)",
    "middle_balanced": "Balanced Compromise",
    "not_improv": "Reference Design",
}

def display_label(representative_key: str) -> str:
    """Label of a representative solution, used in figures, printed summaries and exported '_label' columns."""
    return _DISPLAY_LABELS.get(representative_key, representative_key.replace("_", " ").title())
