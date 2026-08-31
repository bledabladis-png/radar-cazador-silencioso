# -*- coding: utf-8 -*-
"""
tactical_engine.py -- Tactical Score v1.0
Evalua el momentum de corto plazo de cada sector.
"""
import pandas as pd
import numpy as np
from src.utils import get_col
from config.weights import TACTICAL_WEIGHTS

def compute_tactical_score(df_market, sector_etf, benchmark='^GSPC'):
    """Calcula el Tactical Score combinando 5 componentes de corto plazo."""
    try:
        close_sector = get_col(df_market, sector_etf, 'Close')
        close_bench = get_col(df_market, benchmark, 'Close')
        volume_sector = get_col(df_market, sector_etf, 'Volume')
    except KeyError:
        return 0.0

    w = TACTICAL_WEIGHTS

    # RS20
    rs = close_sector / close_bench
    rs20 = rs.pct_change(20).iloc[-1] if len(rs) >= 21 else 0.0
    rs20_norm = np.tanh(rs20 * 10) if pd.notna(rs20) else 0.0

    # Momentum20
    mom20 = close_sector.pct_change(20).iloc[-1] if len(close_sector) >= 21 else 0.0
    mom20_norm = np.tanh(mom20 * 5) if pd.notna(mom20) else 0.0

    # Flujo reciente
    if len(close_sector) >= 6 and len(volume_sector) >= 6:
        ret_5d = close_sector.pct_change(5).iloc[-1]
        vol_5d = volume_sector.iloc[-5:].mean()
        flow_recent = ret_5d * vol_5d / volume_sector.iloc[-10:].mean() if volume_sector.iloc[-10:].mean() > 0 else 0.0
        flow_norm = np.tanh(flow_recent / 2) if pd.notna(flow_recent) else 0.0
    else:
        flow_norm = 0.0

    # Breadth20
    if len(close_sector) >= 20:
        ema20 = close_sector.ewm(span=20, min_periods=20).mean()
        breadth20 = (close_sector.iloc[-20:] > ema20.iloc[-20:]).sum() / 20
        breadth_norm = (breadth20 - 0.5) * 2
    else:
        breadth_norm = 0.0

    # Aceleracion
    if len(close_sector) >= 21:
        mom20_prev = close_sector.pct_change(20).iloc[-6] if len(close_sector) >= 26 else 0.0
        accel = (mom20 - mom20_prev) * 5
        accel_norm = np.tanh(accel * 3) if pd.notna(accel) else 0.0
    else:
        accel_norm = 0.0

    # Composite usando pesos centralizados
    score = (
        w['rs20'] * rs20_norm +
        w['momentum20'] * mom20_norm +
        w['flow_recent'] * flow_norm +
        w['breadth20'] * breadth_norm +
        w['acceleration'] * accel_norm
    )

    return float(np.clip(score, -1, 1))

compute_tactical_score.__doc__ = f"""
Pesos: RS20({TACTICAL_WEIGHTS['rs20']*100:.0f}%), Momentum20({TACTICAL_WEIGHTS['momentum20']*100:.0f}%), Flow({TACTICAL_WEIGHTS['flow_recent']*100:.0f}%), Breadth20({TACTICAL_WEIGHTS['breadth20']*100:.0f}%), Aceleracion({TACTICAL_WEIGHTS['acceleration']*100:.0f}%). Resultado acotado a [-1, +1].
"""
