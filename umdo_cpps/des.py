from __future__ import annotations
import math
from statistics import NormalDist
from typing import Any, Dict, Mapping, Sequence
import numpy as np
from . import config
from .scenarios import EVENT_STATION_MAP, EVENT_TROUGH, EVENT_UV_MAP, get_scenario
from .uncertainty import BASELINE_UV

CONVEYOR_PARAMETERS = {
    1: {"transfer_rate_factor": .70, "parallel_capacity": 1, "power_kW": .15},
    2: {"transfer_rate_factor": 1.00, "parallel_capacity": 1, "power_kW": .35},
    3: {"transfer_rate_factor": 1.18, "parallel_capacity": 2, "power_kW": .45},
    4: {"transfer_rate_factor": 1.43, "parallel_capacity": 3, "power_kW": .60},
}
SCHEDULING_PARAMETERS = {
    1: {"wip_target_min": None, "control_power_kW": .05},
    2: {"wip_target_min": None, "control_power_kW": .08},
    3: {"wip_target_min": 45.0, "control_power_kW": .15},
    4: {"wip_target_min": 30.0, "control_power_kW": .25},
    5: {"wip_target_min": 20.0, "control_power_kW": .40},
}
ENERGY_PARAMETERS = {
    1: {"control_power_kW": .05, "active_factor":1.00, "idle_factor":1.00},
    2: {"control_power_kW": .10, "active_factor":.98, "idle_factor":.60},
    3: {"control_power_kW": .18, "active_factor":.92, "idle_factor":.50},
    4: {"control_power_kW": .30, "active_factor":.88, "idle_factor":.40},
}

# Size of the constraint buffer b1 (CNC -> assembly): it keeps the CNC producing while assembly is disrupted (cobot
# breakdown or operator error, see scenarios.EVENT_STATION_MAP). It covers mean + z * std of the longer of the two
# disruption-duration distributions (uncertainty.BASELINE_UV) at the one-sided protection level
# config.CONSTRAINT_BUFFER_PROTECTION_LEVEL.
def _compute_constraint_buffer_minutes() -> float:
    z = NormalDist().inv_cdf(config.CONSTRAINT_BUFFER_PROTECTION_LEVEL)
    cobot = BASELINE_UV["TTR_Cobot"]
    assembly_error = BASELINE_UV["Tcorr_Assembly"]
    return max(cobot.mean + z * cobot.std, assembly_error.mean + z * assembly_error.std)

CONSTRAINT_BUFFER_MINUTES = _compute_constraint_buffer_minutes()

def build_events(sample: Mapping[str, float]):
    scenario = get_scenario(config.SCENARIO_MODE)
    events = []
    for event_type, start_h in scenario["events"]:
        duration = float(sample[EVENT_UV_MAP[event_type]])
        duration_eff = max(0.0, duration)
        start_min = start_h * 60.0
        end_min = start_min + duration_eff
        events.append({
            "event_type": event_type,
            "station": EVENT_STATION_MAP[event_type],
            "start_min": start_min,
            "end_min": end_min,
            "duration_min": duration,
            "trough": EVENT_TROUGH[event_type],
            "onset_min": config.DISRUPTION_ONSET_MIN,
            "recovery_min": max(config.DISRUPTION_RECOVERY_FLOOR_MIN, config.DISRUPTION_RECOVERY_FRACTION * duration_eff),
        })
    return events

def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)

def _ramp(level_start: float, level_end: float, t: float, t0: float, tau: float) -> float:
    if tau <= 1e-9:
        return level_end if t >= t0 else level_start
    return level_start + (level_end - level_start) * _smoothstep((t - t0) / tau)

def _event_capacity_factor(event: Mapping[str, float], t: float) -> float:
    # After the recovery ramp the capacity returns to nominal (1.0). Production lost during the onset, trough and
    # recovery is not recovered (no catch-up is modelled) and appears as a permanent gap in cumulative production.
    t0, t1 = event["start_min"], event["end_min"]
    onset, recovery = event["onset_min"], event["recovery_min"]
    trough = event["trough"]
    if t < t0:
        return 1.0
    if t < t0 + onset:
        return _ramp(1.0, trough, t, t0, onset)
    if t < t1:
        return trough
    if t < t1 + recovery:
        return _ramp(trough, 1.0, t, t1, recovery)
    return 1.0

def capacity_factor(station: str, t: float, events: Sequence[Mapping[str, float]]) -> float:
    relevant = [e for e in events if e["station"] == station]
    if not relevant:
        return 1.0
    return min(_event_capacity_factor(e, t) for e in relevant)

def _buffer_throttle(level: float, capacity: float) -> float:
    # The downstream draw is tapered over the last config.BUFFER_LOW_THROTTLE_FRACTION of the buffer capacity, so that
    # throughput declines gradually instead of dropping in a single time step when the buffer empties.
    if capacity <= 1e-12:
        return 0.0
    frac = level / capacity
    threshold = config.BUFFER_LOW_THROTTLE_FRACTION
    if frac >= threshold:
        return 1.0
    return _smoothstep(frac / threshold)

def simulate(design: Mapping[str, int], mdo: Mapping[str, float], sample: Mapping[str, float], keep_profile: bool = False):
    nominal_th = max(float(mdo["nominal_throughput"]), 1e-9)
    rate = nominal_th / 60.0
    conveyor = CONVEYOR_PARAMETERS[int(design["v_conv"])]
    scheduling = SCHEDULING_PARAMETERS[int(design["u_sched"])]
    energy = ENERGY_PARAMETERS[int(design["u_energy"])]
    transfer_cap = rate * conveyor["transfer_rate_factor"] * conveyor["parallel_capacity"]

    buffer1_capacity = rate * CONSTRAINT_BUFFER_MINUTES        # b1: CNC -> Assembly (constraint protection)
    buffer2_capacity = rate * config.NON_CRITICAL_BUFFER_MINUTES  # b2: Assembly -> QC (non-critical)
    if config.MODEL_STARTUP_RAMP:
        # Cold start: both buffers start empty. Because the CNC ramps up faster than assembly
        # (CONSTRAINT_STARTUP_RAMP_MIN < STARTUP_RAMP_MIN), b1 builds up a protective stock in the first minutes.
        b1 = 0.0
        b2 = 0.0
    else:
        b1 = buffer1_capacity * config.CONSTRAINT_BUFFER_INITIAL_FRACTION
        b2 = buffer2_capacity * config.NON_CRITICAL_BUFFER_INITIAL_FRACTION
    events = build_events(sample)
    output = 0.0
    energy_kwh = 0.0
    starvation = 0.0
    blockage = 0.0
    profile = []

    total_steps = int(config.SHIFT_HOURS * 60 / config.DT_MIN)
    base_power = float(mdo["nominal_energy_kWh"]) / config.SHIFT_HOURS
    p_cnc, p_ass, p_qc, p_aux = .40*base_power, .30*base_power, .20*base_power, .10*base_power

    for step in range(total_steps):
        t = step * config.DT_MIN
        phi_cnc = capacity_factor("CNC", t, events)
        phi_ass = capacity_factor("ASSEMBLY", t, events)
        phi_qc = capacity_factor("QC", t, events)
        if config.MODEL_STARTUP_RAMP:
            # The CNC (constraint resource) ramps up faster than assembly and QC (see config.py).
            phi_cnc *= _ramp(0.0, 1.0, t, 0.0, config.CONSTRAINT_STARTUP_RAMP_MIN)
            startup_other = _ramp(0.0, 1.0, t, 0.0, config.STARTUP_RAMP_MIN)
            phi_ass *= startup_other
            phi_qc *= startup_other

        total_wip = b1 + b2
        target_min = scheduling["wip_target_min"]
        target = None if target_min is None else rate * target_min
        release_ok = target is None or total_wip < target

        space1 = buffer1_capacity - b1
        cnc_out = phi_cnc * min(rate * config.DT_MIN, max(0.0, space1)) if (release_ok and space1 > 0) else 0.0
        if phi_cnc >= 0.999 and space1 <= 1e-12:
            blockage += config.DT_MIN
        b1 += cnc_out

        space2 = buffer2_capacity - b2
        throttle1 = _buffer_throttle(b1, buffer1_capacity)
        ass_out = phi_ass * throttle1 * min(rate*config.DT_MIN, transfer_cap*config.DT_MIN, max(0.0, space2))
        ass_out = min(ass_out, b1)
        b1 -= ass_out
        b2 += ass_out
        if phi_ass >= 0.999 and ass_out <= 1e-12:
            starvation += config.DT_MIN

        throttle2 = _buffer_throttle(b2, buffer2_capacity)
        qc_out = phi_qc * throttle2 * min(rate*config.DT_MIN, transfer_cap*config.DT_MIN)
        qc_out = min(qc_out, b2)
        b2 -= qc_out
        output += qc_out
        if phi_qc >= 0.999 and qc_out <= 1e-12:
            starvation += config.DT_MIN

        inst_th = qc_out / config.DT_MIN * 60.0
        active = ((p_cnc if cnc_out>0 else 0) + (p_ass if ass_out>0 else 0) + (p_qc if qc_out>0 else 0))
        idle = ((.2*p_cnc if cnc_out<=0 else 0) + (.2*p_ass if ass_out<=0 else 0) + (.2*p_qc if qc_out<=0 else 0))
        energy_kwh += (
            active*energy["active_factor"] + idle*energy["idle_factor"] +
            p_aux + conveyor["power_kW"] + scheduling["control_power_kW"] + energy["control_power_kW"]
        ) * config.DT_MIN / 60.0

        if keep_profile:
            active_events = [e["event_type"] for e in events if e["start_min"] <= t < e["end_min"]]
            profile.append({
                "time_min": t, "time_h": t/60.0,
                "throughput_cartons_h": inst_th,
                "cumulative_cartons": output,
                "buffer_CNC_Assembly": b1,
                "buffer_Assembly_QC": b2,
                "CNC_capacity_factor": phi_cnc,
                "Assembly_capacity_factor": phi_ass,
                "QC_capacity_factor": phi_qc,
                "active_events": "+".join(active_events) if active_events else "NONE",
                "cumulative_energy_kWh": energy_kwh,
            })

    disrupted_th = output / config.SHIFT_HOURS
    return {
        "disrupted_throughput": disrupted_th,
        "energy_kWh": energy_kwh,
        "cartons_shift": output,
        "starvation_min": starvation,
        "blockage_min": blockage,
        "events": events,
        "profile": profile,
    }
