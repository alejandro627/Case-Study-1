from __future__ import annotations
from typing import Any, Dict

# Disruption types and their uncertain durations:
#   CNC_BREAKDOWN     -> TTR_CNC        (CNC station down for repair)
#   COBOT_BREAKDOWN   -> TTR_Cobot      (assembly cobot down for repair)
#   ASSEMBLY_ERROR    -> Tcorr_Assembly (operator error at assembly, correction time)
# The scenarios cover all 7 non-empty combinations of the three events (3 single, 3 double, 1 triple) at fixed start
# times (CNC 3 h, cobot 5 h, assembly 7 h).
SCENARIOS: Dict[int, Dict[str, Any]] = {
    1: {"name": "CNC_BREAKDOWN_ONLY", "events": [("CNC_BREAKDOWN", 3.0)]},
    2: {"name": "COBOT_BREAKDOWN_ONLY", "events": [("COBOT_BREAKDOWN", 5.0)]},
    3: {"name": "ASSEMBLY_ERROR_ONLY", "events": [("ASSEMBLY_ERROR", 7.0)]},
    4: {"name": "CNC_AND_COBOT_BREAKDOWN", "events": [("CNC_BREAKDOWN", 3.0), ("COBOT_BREAKDOWN", 5.0)]},
    5: {"name": "CNC_BREAKDOWN_AND_ASSEMBLY_ERROR", "events": [("CNC_BREAKDOWN", 3.0), ("ASSEMBLY_ERROR", 7.0)]},
    6: {"name": "COBOT_BREAKDOWN_AND_ASSEMBLY_ERROR", "events": [("COBOT_BREAKDOWN", 5.0), ("ASSEMBLY_ERROR", 7.0)]},
    7: {
        "name": "ALL_THREE_DISRUPTIONS",
        "events": [("CNC_BREAKDOWN", 3.0), ("COBOT_BREAKDOWN", 5.0), ("ASSEMBLY_ERROR", 7.0)],
    },
}

EVENT_UV_MAP = {
    "CNC_BREAKDOWN": "TTR_CNC",
    "COBOT_BREAKDOWN": "TTR_Cobot",
    "ASSEMBLY_ERROR": "Tcorr_Assembly",
}
EVENT_STATION_MAP = {
    "CNC_BREAKDOWN": "CNC",
    "COBOT_BREAKDOWN": "ASSEMBLY",
    "ASSEMBLY_ERROR": "ASSEMBLY",
}
# Capacity fraction of the affected station at the trough. Breakdowns stop the station; the operator error reduces the
# capacity of the assembly station.
EVENT_TROUGH = {
    "CNC_BREAKDOWN": 0.0,
    "COBOT_BREAKDOWN": 0.0,
    "ASSEMBLY_ERROR": 0.5,
}

def get_scenario(mode: int):
    if mode not in SCENARIOS:
        raise ValueError("SCENARIO_MODE must be between 1 and 7.")
    return SCENARIOS[mode]
