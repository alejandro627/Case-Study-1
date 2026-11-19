"""Settings of the Case Study 1 UMDA/UMDO (wooden-box CPPS)."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "deterministic_des_mdo_mda_mdf_assistance_no_buffers.py"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHIFT_HOURS = 8.0
DT_MIN = 1.0           # time step of the flow model [min]
EPSILON = 0.02         # TUT threshold: TH(t) < (1 - EPSILON) * TH_nom

SCENARIO_MODE = 7      # disruption scenario (modes 1-7 in scenarios.py); 7 = all three disruptions

# R_phi(t_f) is evaluated R_PHI_OFFSET_MIN after the start of the last disruption. With longer offsets almost every
# sampled disruption is fully recovered before t_f, so that R_phi saturates at 1 and no longer separates designs.
R_PHI_EVALUATION_MODE = "offset_from_last_start"
R_PHI_OFFSET_MIN = 30.0

# Reliability level of the three chance constraints (disrupted throughput, disrupted energy, recovery within the shift).
RELIABILITY_LEVEL = 0.95

# Complementary resilience metrics (PL, TST, TUT, TTR, recovery rate) are computed and exported for every run;
# R_phi(t_f) is the only resilience objective of the UMDO.
# Sustained-recovery criterion for TTR and recovery rate: TH(t) >= RECOVERY_THRESHOLD_FRACTION * TH_nom for all
# t in [t_r, t_end] (independent of EPSILON).
RECOVERY_THRESHOLD_FRACTION = 0.98

# LHS sample sizes: inner loop of the search, high-fidelity re-evaluation, standalone sensitivity analysis.
N_LHS_OPTIMISATION = 200
N_LHS_FINAL = 1500
N_LHS_SENSITIVITY = 500
LHS_SEED = 42

NSGA2_POPULATION = 80
NSGA2_GENERATIONS = 300
NSGA2_SEED = 17
# Seed of the LHS-based initial population (design-variable space), independent of LHS_SEED (uncertain variables)
# and NSGA2_SEED (selection, crossover and mutation).
POPULATION_LHS_SEED = 71

# NSGA-II operators (optimisation.run_nsga2); mutation probability = 1 / number of design variables.
NSGA2_CROSSOVER_PROB = 0.90
NSGA2_CROSSOVER_ETA = 15
NSGA2_MUTATION_ETA = 20

# The ten design variables are integers, whereas SBX and polynomial mutation work on a continuous relaxation.
# RoundingRepair rounds the population after sampling, crossover and mutation in every generation, so that duplicate
# elimination and crowding distance operate on the integer designs.
NSGA2_USE_INTEGER_REPAIR = True

# A run is aborted once this many distinct designs have raised an unexpected evaluation exception (constraint
# infeasibility is not an exception).
NSGA2_FAIL_FAST_THRESHOLD = 5

# Interval (generations) at which the current population is written to a checkpoint file (overwritten each time).
# The checkpoint preserves intermediate results; it is not used to resume a search.
NSGA2_CHECKPOINT_EVERY_N_GEN = 1

STD_MODE = "fixed"  # fixed | half_effect | same_coefficient

# Disruption capacity ramps (smoothstep S-curve): the onset (nominal -> trough) is short and fixed; the recovery
# (trough -> post-recovery level) lasts a fraction of the event duration, with a lower bound for short disruptions.
DISRUPTION_ONSET_MIN = 4.0
DISRUPTION_RECOVERY_FRACTION = 0.20
DISRUPTION_RECOVERY_FLOOR_MIN = 5.0

# Optional shift start-up ramp (off by default): each station ramps from 0 to full capacity from t = 0 with the same
# smoothstep shape. The CNC (constraint resource) ramps faster than the non-constraint stations, which have spare
# capacity. The R_phi window starts at the first disruption, so the ramp durations must stay well below the first
# event time (3 h); PL and TUT include a small start-up contribution when the ramp is on.
MODEL_STARTUP_RAMP = False
CONSTRAINT_STARTUP_RAMP_MIN = 5.0   # CNC
STARTUP_RAMP_MIN = 15.0             # assembly, QC

# Below this fraction of a buffer's capacity the downstream draw is throttled smoothly towards zero instead of
# continuing at full rate until the buffer is empty.
BUFFER_LOW_THROTTLE_FRACTION = 0.30

# Buffers are sized in minutes of each design's own nominal flow, not in absolute units (at 1.5-2.5 cartons/h a fixed
# buffer of 10-20 units would hold several hours of output and mask every disruption).
# b1 (CNC -> assembly) protects the constraint resource (CNC) from blocking when assembly is disrupted; its size is
# computed in des.py from the assembly-side disruption distributions at the protection level below.
# b2 (assembly -> QC) is not critical (both stations have about three times the constraint-limited flow) and has a
# fixed size.
CONSTRAINT_BUFFER_PROTECTION_LEVEL = 0.95
NON_CRITICAL_BUFFER_MINUTES = 10.0
CONSTRAINT_BUFFER_INITIAL_FRACTION = 0.5   # initial fill of b1 when MODEL_STARTUP_RAMP is off
NON_CRITICAL_BUFFER_INITIAL_FRACTION = 0.5 # initial fill of b2 when MODEL_STARTUP_RAMP is off
