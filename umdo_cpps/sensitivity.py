from __future__ import annotations
from typing import Mapping
from . import config
from .analysis import evaluate_umda
from .design import DV_LEVELS
from .uncertainty import UV_ORDER, lhs_unit

def dv_oat(base_design: Mapping[str,int]):
    lhs=lhs_unit(config.N_LHS_SENSITIVITY,len(UV_ORDER),config.LHS_SEED+7000)
    rows=[]
    for dv,levels in DV_LEVELS.items():
        for level in levels:
            design=dict(base_design); design[dv]=level
            summary,_,_=evaluate_umda(design,lhs,keep_profiles=False)
            rows.append({"varied_DV":dv,"level":level,**summary})
    return rows
