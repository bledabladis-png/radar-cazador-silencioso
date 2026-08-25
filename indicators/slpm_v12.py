# -*- coding: utf-8 -*-
"""
slpm_v12.py -- SLPM v1.2 (con ajuste de cobertura y documentacion)
"""
import pandas as pd
import numpy as np
from config.tickers import SECTOR_NAMES
from config.weights import SLPM_WEIGHTS
from config import settings
from config.settings import SLPM_EXPECTED_LEADERS
from indicators.state_machine import classify_leadership_state, get_opportunity_quadrant, validate_state
from indicators.state_transition import confirm_transition

# LIS: métrica de diagnóstico, no señal decisoria.
# Breadth: 0-1 = proporcion de lideres que cumplen condiciones.
# Persistence: 0-1 = proporcion de semanas con senhal positiva.

def compute_leader_breadth_v2(leader_metrics, expected_leaders=SLPM_EXPECTED_LEADERS):
    if not leader_metrics:
        return {'rs_breadth': 0.0, 'momentum_breadth': 0.0, 'flow_breadth': 0.0, 'wyckoff_breadth': 0.0, 'composite': 0.5, 'n_used': 0, 'coverage': 0.0, 'effective_composite': 0.5, 'expected_leaders': expected_leaders}
    n = len(leader_metrics)
    coverage = min(n / expected_leaders, 1.0) if expected_leaders > 0 else 0

    rs_positive = sum(1 for m in leader_metrics if m and (m.get('rs', 1.0) or 1.0) > 1.0) / n
    momentum_positive = sum(1 for m in leader_metrics if m and (m.get('rs_momentum', m.get('rs_mom_20', 0)) or 0) > 0) / n
    flow_positive = sum(1 for m in leader_metrics if m and (m.get('flow_proxy_z', 0) or 0) > 0) / n
    wyckoff_favorable = sum(1 for m in leader_metrics if m and m.get('wyckoff_phase', '') in ('ACCUMULATION', 'MARKUP')) / n

    composite = SLPM_WEIGHTS["leader_breadth"]["rs"] * rs_positive + SLPM_WEIGHTS["leader_breadth"]["momentum"] * momentum_positive + SLPM_WEIGHTS["leader_breadth"]["flow"] * flow_positive + SLPM_WEIGHTS["leader_breadth"]["wyckoff"] * wyckoff_favorable
    effective_composite = composite * coverage if coverage < 0.5 else composite

    return {
        'rs_breadth': rs_positive,
        'momentum_breadth': momentum_positive,
        'flow_breadth': flow_positive,
        'wyckoff_breadth': wyckoff_favorable,
        'composite': composite,
        'effective_composite': effective_composite,
        'n_used': n,
        'expected_leaders': expected_leaders,
        'coverage': coverage,
        'coverage_warning': coverage < 0.5
    }

def compute_leader_integrity(leader_metrics):
    if not leader_metrics:
        return {'lis': 0.0, 'n_leaders': 0}
    scores = []
    for m in leader_metrics:
        if not m:
            continue
        rs = m.get('rs', 1.0)
        if rs is None:
            rs = 1.0
        rs_norm = np.tanh((rs - 1.0) * 2)
        rs_mom = m.get('rs_momentum') or m.get('rs_mom_20', 0) or 0
        mom_norm = np.tanh(rs_mom * 5)
        flow = m.get('flow_proxy_z') or 0
        flow_norm = np.tanh(flow / 2)
        wyckoff_map = {'MARKUP': 1.0, 'ACCUMULATION': 0.75, 'RANGE': 0.0, 'DISTRIBUTION': -0.75, 'MARKDOWN': -1.0}
        wyckoff_score = wyckoff_map.get(m.get('wyckoff_phase', ''), 0.0)
        individual = SLPM_WEIGHTS["lis"]["rs"] * rs_norm + SLPM_WEIGHTS["lis"]["momentum"] * mom_norm + SLPM_WEIGHTS["lis"]["flow"] * flow_norm + SLPM_WEIGHTS["lis"]["wyckoff"] * wyckoff_score
        scores.append(individual)
    if not scores:
        return {'lis': 0.0, 'n_leaders': 0}
    lis_val = np.nanmean(scores) if scores else 0.0
    if np.isnan(lis_val):
        lis_val = 0.0
    return {'lis': float(np.clip(lis_val, -1, 1)), 'n_leaders': len(scores)}

def compute_flow_divergence_v2(leader_metrics, sector_flow_proxy_z, sector_price_flow=None):
    sector_flow_proxy_z = sector_flow_proxy_z if sector_flow_proxy_z is not None else 0.0
    leader_flows = [m.get('flow_proxy_z', np.nan) for m in leader_metrics if m and 'flow_proxy_z' in m]
    valid_leader_flows = [f for f in leader_flows if pd.notna(f)]
    leader_flow_div = float(np.nanmean(valid_leader_flows) - sector_flow_proxy_z) if valid_leader_flows else 0.0
    sector_flow_vs_price_div = float(sector_flow_proxy_z - sector_price_flow) if (sector_price_flow is not None and pd.notna(sector_price_flow)) else 0.0
    leader_flow_std = np.std(valid_leader_flows) if (valid_leader_flows and len(valid_leader_flows) > 1) else 0.0
    structural_flow_div = float(np.nanmean(valid_leader_flows) - leader_flow_std) if valid_leader_flows else 0.0
    composite = 0.50 * leader_flow_div + 0.25 * sector_flow_vs_price_div + 0.25 * structural_flow_div
    return {'leader_flow_div': leader_flow_div, 'sector_flow_vs_price_div': sector_flow_vs_price_div, 'structural_flow_div': structural_flow_div, 'composite': composite}

def evaluate_slpm_v12(df_market, sector_results, leader_metrics, top_sector_flow,
                       tactical_scores=None, structural_scores=None, sector_persistence=None):
    ranking = sector_results.get('ranking', [])
    if not ranking:
        return {'sector': '', 'sector_etf': '', 'state': 'UNRESOLVED', 'opportunity_quadrant': 'Transition'}
    sector_etf = ranking[0][0]
    sector_name = SECTOR_NAMES.get(sector_etf, sector_etf)

    breadth_v2 = compute_leader_breadth_v2(leader_metrics, expected_leaders=SLPM_EXPECTED_LEADERS)
    integrity = compute_leader_integrity(leader_metrics)
    top_sector_flow = top_sector_flow if top_sector_flow is not None else 0.0
    flow_div_v2 = compute_flow_divergence_v2(leader_metrics, top_sector_flow, None)

    tactical_score = tactical_scores.get(sector_etf, 0.0) if tactical_scores else 0.0
    structural_score = structural_scores.get(sector_etf, 0.0) if structural_scores else 0.0
    persistence = sector_persistence.get(sector_etf) if sector_persistence else None
    if persistence is None:
        persistence = 0.5
        print(f"    SLPM v1.2: Persistence no disponible para {sector_name}, usando 0.5 neutro.")

    effective_breadth = breadth_v2['effective_composite']

    result = classify_leadership_state(tactical_score, structural_score, effective_breadth, persistence, coverage=breadth_v2['coverage'])
    instant_state = result['state']
    instant_reason = result['reason']
    if not leader_metrics:
        sector_phase = ranking[0][3] if len(ranking[0]) > 3 else 'desconocida'
        instant_reason += f" El sector lider ({sector_name}) esta en fase {sector_phase}, lo que impide calcular metricas de lideres."
    instant_reason_code = result.get('reason_code', 'UNKNOWN')

    transition_data = confirm_transition(instant_state)
    state = transition_data['confirmed_state']
    reason = instant_reason
    reason_code = instant_reason_code
    quadrant = get_opportunity_quadrant(state)

    errors = validate_state(state, tactical_score, structural_score, effective_breadth, persistence)
    if errors:
        print(f"    WARN SLPM v1.2 STATE VALIDATION FAILED para {sector_name}:")
        for e in errors:
            print(f"      - {e}")

    return {
        'sector': sector_name, 'sector_etf': sector_etf,
        'state': state, 'state_reason': reason,
        'state_reason_code': reason_code,
        'instant_state': instant_state,
        'previous_state': transition_data['previous_state'],
        'transition': transition_data['transition'],
        'consecutive_count': transition_data['consecutive_count'],
        'opportunity_quadrant': quadrant,
        'leader_breadth_v2': breadth_v2,
        'leader_integrity': integrity,
        'flow_divergence_v2': flow_div_v2,
        'tactical_score': tactical_score,
        'structural_score': structural_score,
        'persistence': persistence,
        'top_sector_flow': top_sector_flow,
        'input_scores': {
            'tactical': tactical_score, 'structural': structural_score,
            'lis': integrity['lis'], 'breadth': breadth_v2['composite'],
            'effective_breadth': effective_breadth, 'persistence': persistence
        },
        'validation_errors': errors,
        'data_quality': result.get('data_quality', 'UNKNOWN'),
        'narrow_leadership': (
            flow_div_v2.get('leader_flow_div', 0) > 0.2 and
            top_sector_flow < 0.2 and
            len(leader_metrics) >= 3
        )
    }
