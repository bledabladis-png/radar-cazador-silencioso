# -*- coding: utf-8 -*-
"""
structural_engine.py -- Structural Score v1.0
Evalua la fortaleza estructural de largo plazo de cada sector.
"""
import pandas as pd
import numpy as np
from src.utils import get_col
from config import settings
from config.weights import STRUCTURAL_WEIGHTS

def compute_structural_score(df_market, sector_etf, leader_breadth=0.5, flow_structure=0.0, persistence=0.5, benchmark='^GSPC'):
    """Calcula el Structural Score de largo plazo.
    Pesos: RS multi-ventana 63/126/252d (50%), Flow Structure (30%),
    Persistence (20%). Resultado acotado a [-1, +1]."""
    try:
        close_sector = get_col(df_market, sector_etf, 'Close')
        close_bench = get_col(df_market, benchmark, 'Close')
    except KeyError:
        return 0.0

    w = STRUCTURAL_WEIGHTS
    rs = close_sector / close_bench

    def rs_momentum(window):
        if len(rs) < window:
            return 0.0
        return (rs.iloc[-1] / rs.iloc[-window] - 1) if rs.iloc[-window] != 0 else 0.0

    rs63 = rs_momentum(settings.RS_MEDIUM_WINDOW)
    rs126 = rs_momentum(settings.MOMENTUM_LONG_WINDOW)
    rs252 = rs_momentum(settings.RS_STRUCTURAL_WINDOW)

    rs_values = [rs63, rs126, rs252]
    rs_structural = np.mean([v for v in rs_values if pd.notna(v)]) if rs_values else 0.0
    rs_norm = np.tanh(rs_structural * 2) if pd.notna(rs_structural) else 0.0

    flow_norm = np.tanh(flow_structure)
    pers_norm = (persistence - 0.5) * 2

    score = (
        w['rs_structural'] * rs_norm +
        w['flow_structure'] * flow_norm +
        w['persistence'] * pers_norm
    )

    return float(np.clip(score, -1, 1))
