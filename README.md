# Case Study 1 – Resilience-oriented design of a wooden-box CPPS

Python implementation of Case Study 1 of the doctoral thesis titled "Sustainability-oriented optimization of modular and reconfigurable cyber-physical production systems: A Complexity-engineering-based framework for resilience". The case study applies a
complexity-engineering-based design framework to a virtual cyber-physical production system (CPPS) that produces
wooden boxes in three stages: CNC laser machining of plywood panels, collaborative (cobot) assembly, and quality
control and packaging (two boxes per carton) over an 8-hour shift.

| Step | Content | Code |
|---|---|---|
| 1. AD complexity framework | functional decomposition (documented in the thesis); defines the design variables | – |
| 2. Uncertain system modelling | baseline SimPy DES, three-discipline MDA solved with an MDF architecture, deterministic optimisation, uncertain disruption durations and scenarios | `baseline_des_wooden_box.py`, `deterministic_des_mdo_mda_mdf_assistance_no_buffers.py`, `mdo_mdf_optimizer_no_buffers.py`, `umdo_cpps/uncertainty.py`, `umdo_cpps/scenarios.py` |
| 3. UMDA | uncertainty propagation with Latin hypercube sampling, sensitivity analysis, LHS convergence | `umdo_cpps/analysis.py`, `umdo_cpps/disruption_sensitivity.py`, `umdo_cpps/lhs_convergence.py`, `umdo_cpps/metrics.py` |
| 4. UMDO | constrained three-objective NSGA-II with high-fidelity re-evaluation of the Pareto candidates | `umdo_cpps/optimisation.py`, `umdo_cpps/main.py` |

## Model overview

* **Deterministic model.** Three coupled disciplines (CNC, assembly, QC/packaging) exchange the flows q12
  (CNC → assembly) and q23 (assembly → QC); the multidisciplinary analysis converges these couplings (MDF), and each
  design is evaluated for throughput [cartons/h], energy [kWh/shift], WIP and cycle time.
* **Design variables (291,600 configurations).** Assistance at the CNC, assembly and QC stations (a_CNC, a_A, a_QC),
  flexibility of assembly and QC (r_A, r_QC), conveyor/transfer speed (v_conv), scheduling (u_sched), maintenance
  (u_maint), inspection (u_inspect) and energy-management (u_energy) policies.
* **Uncertainty.** Repair time of the CNC (TTR_CNC, 45 ± 15 min), repair time of the cobot (TTR_Cobot, 35 ± 12 min) and
  correction time of an assembly operator error (Tcorr_Assembly, 18 ± 6 min), scaled by design-dependent coefficients.
* **Disruption scenarios.** All seven combinations of a CNC breakdown (3 h), a cobot breakdown (5 h) and an assembly
  error (7 h); the UMDO uses scenario 7 (all three disruptions).
* **Disrupted performance.** A time-stepped flow model (1-min resolution) with buffers between the stations and
  capacity ramps at the onset and recovery of each disruption.
* **Objectives and constraints.** Maximise mean disrupted throughput, minimise mean energy and maximise mean
  R_phi(t_f) (evaluated 30 min after the start of the last disruption), subject to six deterministic engineering
  constraints and three chance constraints (disrupted throughput, disrupted energy and recovery within the shift,
  reliability 0.95). PL, TST, TUT, TTR and recovery rates are reported as complementary resilience metrics.

## Repository structure

```
baseline_des_wooden_box.py                              baseline SimPy DES of the line (reference configuration)
deterministic_des_mdo_mda_mdf_assistance_no_buffers.py  deterministic MDA/MDF model and one-at-a-time sweep
mdo_mdf_optimizer_no_buffers.py                         deterministic bi-objective optimisation (throughput, energy)
run_umdo.py                                             UMDA/UMDO workflow (settings in umdo_cpps/config.py)
run_disruption_sensitivity.py                           LHS sensitivity of each disruption, alone and combined
run_lhs_convergence_study.py                            LHS sample-size convergence
run_baseline_vs_optimized.py                            baseline vs representative Pareto designs
run_hypervolume_study.py                                hypervolume convergence of NSGA-II
umdo_cpps/                                              UMDA/UMDO package and tests
results/deterministic/                                  outputs of the deterministic model
results/umdo_scenario_7_reference_run/                  UMDO results for scenario 7 (population 80, 300 generations)
```

## Installation

Python 3.10 or later (developed with Python 3.12).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

## Running the case study

All commands are run from the repository root.

```bash
python baseline_des_wooden_box.py                                # baseline DES, 8-h shift
python deterministic_des_mdo_mda_mdf_assistance_no_buffers.py    # deterministic MDA/MDF and one-at-a-time sweep
python mdo_mdf_optimizer_no_buffers.py                           # deterministic optimisation (Pareto front)

python run_disruption_sensitivity.py                             # UMDA: disruption sensitivity (N = 500)
python run_lhs_convergence_study.py                              # UMDA: LHS convergence
python run_umdo.py                                               # UMDO (NSGA-II, N = 200 in the search, N = 1500 re-evaluation)
python run_baseline_vs_optimized.py                              # baseline vs representative designs
python run_hypervolume_study.py                                  # hypervolume vs generation
```

The main settings are in `umdo_cpps/config.py`: scenario (`SCENARIO_MODE`, 7 = all three disruptions), LHS sample
sizes and seeds (`N_LHS_OPTIMISATION` = 200, `N_LHS_FINAL` = 1500, `N_LHS_SENSITIVITY` = 500, `LHS_SEED` = 42),
NSGA-II settings (population 80, 300 generations, seed 17) and the R_phi evaluation time (30 min after the start of
the last disruption). `run_umdo.py` writes its outputs to a new folder `results/<label>_<timestamp>/` and never
overwrites earlier runs.

Tests:

```bash
python -m umdo_cpps.test_resilience_metrics
python -m umdo_cpps.test_nsga2_discrete
```

## Results

`results/umdo_scenario_7_reference_run/` contains the UMDO results for scenario 7: the high-fidelity feasible Pareto
front (`pareto_front_*.csv`, 79 designs), all re-evaluated candidates, the uncertainty propagation of the
representative designs, the uncertain-variable and design-variable sensitivity analyses, the resilience-over-time
profiles and the figures. `important_results_*.xlsx` summarises the representative designs.

## License and citation

The code is released under the MIT License (see `LICENSE`).

If you use this code, please cite: Humberto Alejandro Barrero arciniegas, "Sustainability-oriented optimization of modular and reconfigurable cyber-physical production systems: A Complexity-engineering-based framework for resilience", doctoral thesis, Free University of Bozen-Bolzano, 2026.
