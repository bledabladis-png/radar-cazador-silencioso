# -*- coding: utf-8 -*-
"""
structural_leadership.py -- SLPM v1.0 (Pre-Validation)
Structural Leadership Persistence Module
Evalúa la calidad estructural del liderazgo sectorial actual.
"""
import pandas as pd
import numpy as np
from src.utils import get_col, robust_zscore

def calculate_structural_rs(df_market, sector_etf, benchmark='^GSPC'):
    """
    Structural Relative Strength combinando horizontes 20, 63, 126 y 252 días.
    """
    try:
        close_sector = get_col(df_market, sector_etf, 'Close')
        close_bench = get_col(df_market, benchmark, 'Close')
    except KeyError:
        return None

    rs = close_sector / close_bench

    def rs_momentum(window):
        if len(rs) < window:
            return 0.0
        return (rs.iloc[-1] / rs.iloc[-window] - 1) if rs.iloc[-window] != 0 else 0.0

    rs20 = rs_momentum(20)
    rs63 = rs_momentum(63)
    rs126 = rs_momentum(126)
    rs252 = rs_momentum(252)

    weights = np.array([0.15, 0.20, 0.30, 0.35])
    values = np.array([rs20, rs63, rs126, rs252])
    mask = np.isfinite(values)

    if mask.sum() < 3:
        return 0.0

    return float(np.sum(values[mask] * weights[mask] / weights[mask].sum()))


def calculate_leader_flow_divergence(leader_metrics, sector_flow_z):
    """
    Diferencia entre el flujo medio de los líderes y el flujo agregado del sector.
    """
    leader_flows = [m.get('flow_z', np.nan) for m in leader_metrics if m and 'flow_z' in m]
    valid_flows = [f for f in leader_flows if pd.notna(f)]

    if not valid_flows:
        return 0.0

    return float(np.mean(valid_flows) - sector_flow_z)


def calculate_leader_breadth(leader_metrics):
    """
    Porcentaje de líderes en fases estructuralmente favorables (MARKUP o ACCUMULATION).
    """
    valid_phases = [m.get('wyckoff_phase', '') for m in leader_metrics if m and 'wyckoff_phase' in m]
    if not valid_phases:
        return 0.5

    positive = sum(1 for p in valid_phases if p in ('MARKUP', 'ACCUMULATION'))
    return positive / len(valid_phases)


def calculate_tactical_score(sector_metrics):
    """
    Score táctico basado en RS20 y flujo sectorial.
    """
    rs20 = sector_metrics.get('rs_mom_20', 0)
    flow_z = sector_metrics.get('flow_z', 0)

    if not np.isfinite(rs20) or not np.isfinite(flow_z):
        return 0.0

    return float(0.5 * np.tanh(rs20 * 10) + 0.5 * np.tanh(flow_z))


def calculate_structural_score(struct_rs, leader_breadth, flow_divergence):
    """
    Score estructural combinando RS, amplitud y divergencia de flujo.
    Componentes normalizados a escala comparable antes de combinar.
    """
    # Normalizar cada componente a [-1, 1] con tanh
    struct_rs_norm = np.tanh(struct_rs * 5) if struct_rs is not None else 0.0
    leader_breadth_norm = (leader_breadth - 0.5) * 2  # [0,1] -> [-1,1]
    lfd_norm = np.tanh(flow_divergence * 2) if flow_divergence is not None else 0.0

    weights = np.array([0.30, 0.40, 0.30])
    values = np.array([struct_rs_norm, leader_breadth_norm, lfd_norm])

    return float(np.sum(values * weights))


def classify_leadership(structural_score, tactical_score, leader_breadth):
    """
    Clasifica el estado del liderazgo en 3 estados.
    """
    if structural_score > 0.20 and leader_breadth >= 0.4:
        return 'LEADERSHIP_CONFIRMED'
    elif structural_score < -0.20 and leader_breadth < 0.4:
        return 'STRUCTURAL_DETERIORATION'
    else:
        return 'TACTICAL_CORRECTION'


def evaluate_slpm(df_market, sector_results, leader_metrics, sector_flow_z):
    """
    Función principal del SLPM.
    Evalúa el liderazgo estructural de los sectores líderes.
    """
    if not sector_results or 'ranking' not in sector_results:
        return None

    top_sector = sector_results['ranking'][0]  # (ticker, name, score, wyckoff)
    sector_etf = top_sector[0]
    sector_name = top_sector[1]
    sector_score = top_sector[2]
    sector_wyckoff = top_sector[3]

    struct_rs = calculate_structural_rs(df_market, sector_etf)

    flow_divergence = calculate_leader_flow_divergence(leader_metrics, sector_flow_z)

    leader_breadth = calculate_leader_breadth(leader_metrics)

    sector_metrics_for_tactical = {
        'rs_mom_20': 0,
        'flow_z': sector_flow_z
    }
    tactical_score = calculate_tactical_score(sector_metrics_for_tactical)

    structural_score = calculate_structural_score(
        struct_rs,
        leader_breadth,
        flow_divergence
    )

    state = classify_leadership(structural_score, tactical_score, leader_breadth)

    return {
        'sector': sector_name,
        'sector_etf': sector_etf,
        'sector_score': sector_score,
        'sector_wyckoff': sector_wyckoff,
        'struct_rs': struct_rs,
        'flow_divergence': flow_divergence,
        'leader_breadth': leader_breadth,
        'tactical_score': tactical_score,
        'structural_score': structural_score,
        'state': state
    }
