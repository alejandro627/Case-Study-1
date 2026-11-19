from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from . import config

def _sustained_recovery_index(values: np.ndarray, threshold: float, start_index: int, end_index: Optional[int] = None) -> Optional[int]:
    """First index i in [start_index, end_index] such that values[i:end_index + 1] are all >= threshold, i.e. the
        trajectory stays at or above the threshold for the rest of the window. Returns None if there is no such index;
        end_index defaults to the last index.
    """
    end_index = len(values) - 1 if end_index is None else end_index
    for i in range(start_index, end_index + 1):
        if np.all(values[i:end_index + 1] >= threshold):
            return i
    return None

def _clamp_idx(t: float, n: int) -> int:
    return int(min(n - 1, max(0, round(t / config.DT_MIN))))

def calculate(profile: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, float]], nominal_th: float) -> Dict[str, Any]:
    if not profile:
        raise ValueError("A time profile is required to compute resilience metrics.")
    values = np.asarray([float(row["throughput_cartons_h"]) for row in profile])
    times = np.asarray([float(row["time_min"]) for row in profile])
    n = len(values)
    dt_h = config.DT_MIN / 60.0

    # ---- PL and TUT ----
    pl = float(np.sum(np.maximum(0.0, nominal_th - values)) * dt_h)
    tut_threshold = (1.0 - config.EPSILON) * nominal_th
    tut = float(np.sum(values < tut_threshold) * config.DT_MIN)

    # ---- shared timing anchors ----
    event_starts = [float(e["start_min"]) for e in events] or [0.0]
    event_ends = [float(e["end_min"]) for e in events] or [0.0]
    t_e_first = min(event_starts)
    last_start = max(event_starts)
    last_end = max(event_ends)
    shift_end_min = config.SHIFT_HOURS * 60.0
    first_idx = _clamp_idx(t_e_first, n)
    last_end_idx = _clamp_idx(last_end, n)
    recovery_threshold = config.RECOVERY_THRESHOLD_FRACTION * nominal_th

    # ---- TST: measured from the end of the last disruption (last_end) with the (1 - EPSILON) threshold ----
    tst_r_idx = _sustained_recovery_index(values, tut_threshold, last_end_idx)
    tst = float(times[tst_r_idx] - last_end) if tst_r_idx is not None else shift_end_min

    # TTR, first-to-final span: from the onset of the first disruption to the final sustained-recovery point. For
    # multi-event scenarios this is a span, not a per-disruption recovery time (see total_recovery_interval_min).
    # Reported only; R_phi(t_f) is the only resilience objective.
    r_idx = _sustained_recovery_index(values, recovery_threshold, first_idx)
    recovered = r_idx is not None
    if recovered:
        t_r = float(times[r_idx])
        ttr_ftf_min = t_r - t_e_first
        ttr_ftf_censored = False
        ttr_ftf_followup_min = float("nan")
    else:
        t_r = None
        ttr_ftf_min = float("nan")
        ttr_ftf_censored = True
        ttr_ftf_followup_min = float(shift_end_min - t_e_first)

    search_slice = values[first_idx:]
    d_idx = first_idx + int(np.argmin(search_slice))
    t_d = float(times[d_idx])
    phi_d = float(values[d_idx])
    if recovered and t_r > t_d + 1e-9:
        phi_r = float(values[r_idx])
        dt_h_rec = (t_r - t_d) / 60.0
        rr_norm_ftf = ((phi_r / nominal_th) - (phi_d / nominal_th)) / dt_h_rec
        rr_th_ftf = (phi_r - phi_d) / dt_h_rec
    else:
        rr_norm_ftf = float("nan")
        rr_th_ftf = float("nan")

    # ---- R_phi(t_f) and R_phi(end of shift) ----
    if config.R_PHI_EVALUATION_MODE == "end_of_shift":
        tf = shift_end_min - config.DT_MIN
    else:
        tf = min(shift_end_min - config.DT_MIN, last_start + config.R_PHI_OFFSET_MIN)
    tf_idx = _clamp_idx(tf, n)
    phi_td = float(np.min(values[first_idx:tf_idx + 1])) if tf_idx >= first_idx else nominal_th
    phi_tf = float(values[tf_idx])
    denom = nominal_th - phi_td
    r_tf = 1.0 if denom <= 1e-12 else max(0.0, min(1.0, (phi_tf - phi_td) / denom))
    phi_end = float(values[-1])
    r_end = 1.0 if denom <= 1e-12 else max(0.0, min(1.0, (phi_end - phi_td) / denom))

    # Episode-level recovery analysis and two scenario-level TTR definitions (reported only).
    # Disruptions are grouped into episodes by sustained throughput recovery: if the trajectory has not recovered
    # before the next disruption starts, the next disruption joins the same (compound) episode, and no event-specific
    # TTR is assigned to its constituents (status 'overlapping_superseded'). The recovery window of an episode runs
    # from its onset to the onset of the next episode (or the end of the shift), so windows do not overlap. Episode
    # statuses: 'recovered_before_next_disruption', 'overlapping_superseded',
    # 'not_recovered_before_next_disruption_or_horizon' (last episode only) and 'not_identifiable' (zero-length
    # window).
    # - total_recovery_interval_min: duration of the union of the episode intervals [onset, recovery or window end],
    # right-censored at the window boundary for non-identifiable episodes (not an MTTR).
    # - TTR_event_mean_min: mean of the identifiable episode-level TTRs only.
    sorted_events = sorted(events, key=lambda e: float(e["start_min"]))

    # Episode construction: starting at event k, search for sustained recovery in [start_k, start_k+1). If none is
    # found, event k+1 joins the episode and the window is extended to [start_k, start_k+2), until recovery is found
    # or the events run out (right-censored at the end of the shift).
    episodes: List[Dict[str, Any]] = []
    cursor = 0
    while cursor < len(sorted_events):
        constituents = [sorted_events[cursor]]
        ep_start = float(sorted_events[cursor]["start_min"])
        next_idx = cursor + 1
        while True:
            window_end = float(sorted_events[next_idx]["start_min"]) if next_idx < len(sorted_events) else shift_end_min
            ep_start_idx_probe = _clamp_idx(ep_start, n)
            window_end_idx_probe = _clamp_idx(window_end, n)
            found = (
                _sustained_recovery_index(values, recovery_threshold, ep_start_idx_probe, window_end_idx_probe - 1)
                if window_end_idx_probe - 1 >= ep_start_idx_probe else None
            )
            if found is not None or next_idx >= len(sorted_events):
                break
            constituents.append(sorted_events[next_idx])
            next_idx += 1
        episodes.append({"events": constituents, "start": ep_start})
        cursor = next_idx

    event_ttr_flat: Dict[str, Any] = {}
    union_intervals: List[Tuple[float, float]] = []
    identifiable_episode_ttrs: List[float] = []
    identifiable_episode_rates: List[float] = []  # subset of identifiable episodes that ALSO have a defined (non-NaN) rate - see "undefined rate" cases below (normalized, h^-1)
    identifiable_episode_rates_nonnorm: List[float] = []  # same subset, absolute units (cartons/h^2)
    status_counts: Dict[str, int] = {}

    for i, ep in enumerate(episodes):
        ep_start = ep["start"]
        next_start = episodes[i + 1]["start"] if i + 1 < len(episodes) else shift_end_min
        ep_start_idx = _clamp_idx(ep_start, n)
        window_end_idx = _clamp_idx(next_start, n)

        if window_end_idx - 1 >= ep_start_idx:
            local_slice = values[ep_start_idx:window_end_idx]
            local_d_idx = ep_start_idx + int(np.argmin(local_slice))
            idx = _sustained_recovery_index(values, recovery_threshold, ep_start_idx, window_end_idx - 1)
        else:
            local_d_idx = ep_start_idx
            idx = None
        t_d_i = float(times[local_d_idx])
        phi_d_i = float(values[local_d_idx])
        identifiable = idx is not None

        if window_end_idx - 1 < ep_start_idx:
            status = "not_identifiable"  # degenerate/zero-length window
        elif identifiable:
            status = "recovered_before_next_disruption"
        else:
            status = "not_recovered_before_next_disruption_or_horizon"

        if identifiable:
            t_r_i = float(times[idx])
            ttr_i = t_r_i - ep_start
            if t_r_i > t_d_i + 1e-9:
                phi_r_i = float(values[idx])
                # rate_i is the normalised recovery rate; rate_i_nonnorm is the same rate in absolute units
                # (cartons/h^2), computed independently.
                rate_i = ((phi_r_i / nominal_th) - (phi_d_i / nominal_th)) / ((t_r_i - t_d_i) / 60.0)
                rate_i_nonnorm = 60.0 * (phi_r_i - phi_d_i) / (t_r_i - t_d_i)  # cartons/h^2
            else:
                rate_i = float("nan")
                rate_i_nonnorm = float("nan")
            interval_end = t_r_i
            identifiable_episode_ttrs.append(ttr_i)
            if not np.isnan(rate_i):
                identifiable_episode_rates.append(rate_i)
                identifiable_episode_rates_nonnorm.append(rate_i_nonnorm)
        else:
            t_r_i = float("nan")
            ttr_i = float("nan")
            rate_i = float("nan")
            rate_i_nonnorm = float("nan")
            interval_end = next_start  # right-censored at this episode's own window boundary

        union_intervals.append((ep_start, interval_end))
        status_counts[status] = status_counts.get(status, 0) + 1

        is_compound = len(ep["events"]) > 1
        for e in ep["events"]:
            key_base = f"event_TTR_{e['event_type']}"
            event_status = "overlapping_superseded" if is_compound else status
            event_ttr_flat[f"{key_base}_min"] = float("nan") if is_compound else ttr_i
            event_ttr_flat[f"{key_base}_identifiable"] = False if is_compound else identifiable
            event_ttr_flat[f"{key_base}_status"] = event_status
            event_ttr_flat[f"{key_base}_t_e_min"] = float(e["start_min"])
            event_ttr_flat[f"{key_base}_end_min"] = float(e["end_min"])
            event_ttr_flat[f"{key_base}_t_d_min"] = float("nan") if is_compound else t_d_i
            event_ttr_flat[f"{key_base}_t_r_min"] = float("nan") if is_compound else t_r_i
            event_ttr_flat[f"{key_base}_recovery_rate_norm_per_h"] = float("nan") if is_compound else rate_i
            event_ttr_flat[f"{key_base}_recovery_rate_TH_per_h2"] = float("nan") if is_compound else rate_i_nonnorm
            event_ttr_flat[f"{key_base}_part_of_compound_episode"] = is_compound
            event_ttr_flat[f"{key_base}_episode_index"] = i
            # The compound-episode fields are always written (NaN or '' when the event is not part of a compound
            # episode) so that every realisation has the same set of keys.
            event_ttr_flat[f"{key_base}_compound_TTR_min"] = ttr_i if is_compound else float("nan")
            event_ttr_flat[f"{key_base}_compound_status"] = status if is_compound else ""
            event_ttr_flat[f"{key_base}_compound_recovery_rate_norm_per_h"] = rate_i if is_compound else float("nan")
            event_ttr_flat[f"{key_base}_compound_recovery_rate_TH_per_h2"] = rate_i_nonnorm if is_compound else float("nan")

    union_intervals.sort()
    merged: List[List[float]] = []
    for s, e_ in union_intervals:
        if merged and s <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], e_)
        else:
            merged.append([s, e_])
    total_recovery_interval_min = float(sum(e_ - s for s, e_ in merged))
    n_episodes = len(episodes)
    n_identifiable_episodes = len(identifiable_episode_ttrs)
    total_recovery_interval_censored = n_identifiable_episodes < n_episodes
    ttr_event_mean_min = float(np.mean(identifiable_episode_ttrs)) if identifiable_episode_ttrs else float("nan")

    # Mean recovery rate of this realisation over its identifiable episodes with a defined rate (a rate is undefined,
    # e.g., for a zero-length t_d -> t_r interval). analysis.py aggregates it over the realisations
    # (realisation-weighted), separately from the episode-weighted (pooled) aggregate.
    # The average recovery rate is the net increase in throughput per unit time between the point of maximum
    # degradation (t_d) and sustained recovery (t_r) of one episode. It differs from the first-to-final rate, which is
    # a single slope from the global trough to the final recovery point over the whole shift.
    episode_recovery_rate_mean_norm_per_h = float(np.mean(identifiable_episode_rates)) if identifiable_episode_rates else float("nan")
    episode_recovery_rate_mean_TH_per_h2 = float(np.mean(identifiable_episode_rates_nonnorm)) if identifiable_episode_rates_nonnorm else float("nan")
    n_rate_defined = len(identifiable_episode_rates)
    n_rate_undefined_among_identifiable = n_identifiable_episodes - n_rate_defined

    return {
        "PL_cartons": pl,
        "TST_min": tst,
        "TUT_min": tut,
        "R_phi_tf": r_tf,
        "R_phi_end_shift": r_end,
        "R_phi_tf_time_min": tf,
        "phi_td": phi_td,
        "phi_tf": phi_tf,
        "t_e_min": t_e_first,

        # ---- first-to-final recovery span (reported only) ----
        "recovered_within_horizon": recovered,
        "TTR_first_to_final_min": ttr_ftf_min,
        "TTR_first_to_final_censored": ttr_ftf_censored,
        "TTR_first_to_final_followup_min": ttr_ftf_followup_min,
        "t_d_first_to_final_min": t_d,
        "phi_d_first_to_final": phi_d,
        "t_r_first_to_final_min": t_r if recovered else float("nan"),
        "recovery_rate_norm_per_h_first_to_final": rr_norm_ftf,
        "recovery_rate_TH_per_h2_first_to_final": rr_th_ftf,

        # ---- total recovery-interval duration (union of episode intervals) ----
        "total_recovery_interval_min": total_recovery_interval_min,
        "total_recovery_interval_censored": total_recovery_interval_censored,
        "total_recovery_interval_n_episodes": n_episodes,
        "total_recovery_interval_n_identifiable": n_identifiable_episodes,

        # ---- mean event/episode-level TTR ----
        "TTR_event_mean_min": ttr_event_mean_min,
        "TTR_event_mean_n_identifiable": n_identifiable_episodes,
        "TTR_event_mean_n_recovered": status_counts.get("recovered_before_next_disruption", 0),
        "TTR_event_mean_n_overlapping_superseded": sum(1 for ep in episodes if len(ep["events"]) > 1),
        "TTR_event_mean_n_not_recovered": status_counts.get("not_recovered_before_next_disruption_or_horizon", 0),
        "TTR_event_mean_n_not_identifiable": status_counts.get("not_identifiable", 0),

        # ---- mean episode-level average recovery rate (R_rate) ----
        "episode_recovery_rate_mean_norm_per_h": episode_recovery_rate_mean_norm_per_h,
        "episode_recovery_rate_mean_TH_per_h2": episode_recovery_rate_mean_TH_per_h2,
        "episode_recovery_rate_n_defined": n_rate_defined,
        "episode_recovery_rate_n_undefined_among_identifiable": n_rate_undefined_among_identifiable,

        # ---- per-event breakdown ----
        **event_ttr_flat,
    }
