# -*- coding: utf-8 -*-
"""slpm_v12.py -- SLPM v1.2 (con ajuste de cobertura y documentacion).

A3.3-08 (2026-09-28): antes la funcion _safe_mean estaba definida entre
el coding line y el docstring del modulo. Efecto: (a) slpm_v12.__doc__
era None (el string de triple comilla no era el primer statement), (b)
la funcion referenciaba pd/np importados mas abajo. El contrato seguia
funcionando por late-binding de Python, pero la estructura era
enganosa. Reordenado: docstring primero, luego imports, luego helpers.
"""
import pandas as pd
from src.utils import safe_std
import numpy as np
from config.tickers import SECTOR_NAMES
from config.weights import SLPM_WEIGHTS
from config.settings import SLPM_EXPECTED_LEADERS
from indicators.state_machine import classify_leadership_state, get_opportunity_quadrant, validate_state
from indicators.state_transition import confirm_transition


def _safe_mean(values):
    """Media de valores válidos; 0.0 si no hay ninguno."""
    if not values:
        return 0.0
    valid = [v for v in values if pd.notna(v)]
    return float(np.mean(valid)) if valid else 0.0

# LIS: métrica de diagnóstico, no señal decisoria.
# Breadth: 0-1 = proporcion de lideres que cumplen condiciones.
# Persistence: 0-1 = proporcion de semanas con senhal positiva.

_VALID_WYCKOFF_PHASES = frozenset({
    "MARKUP", "ACCUMULATION", "RANGE", "DISTRIBUTION", "MARKDOWN",
})


def _safe_float(v):
    """Conversion segura a float finito. Devuelve None si no aplica.

    N1 ronda 2 (dictamen C1+C8, 2026-10-14): None, NaN, +Inf, -Inf,
    strings no numericas, listas, dicts, objetos y OverflowError
    cuentan como dato ausente.
    """
    if v is None:
        return None
    try:
        vf = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(vf):
        return None
    return vf


def _breadth_unavailable(status, expected_leaders):
    """Dict de leader_breadth_v2 para casos sin datos calculables.

    status en {"INPUT_UNAVAILABLE", "NO_LEADERS"}.
    N1 ronda 2 (dictamen C4+C9): ningun campo numerico se fabrica a
    0.0; todos van a NaN. coverage_warning True si expected_leaders>0.
    """
    nan = float("nan")
    return {
        "rs_breadth": nan,
        "rs_breadth_status": status,
        "rs_breadth_missing": 0,
        "leader_breadth_status": status,
        "momentum_breadth": nan,
        "momentum_breadth_missing": 0,
        "flow_breadth": nan,
        "flow_breadth_missing": 0,
        "wyckoff_breadth": nan,
        "wyckoff_breadth_missing": 0,
        "composite": nan,
        "effective_composite": nan,
        "n_used": 0,
        "expected_leaders": expected_leaders,
        "coverage": 0.0,
        "coverage_warning": expected_leaders > 0,
    }


def compute_leader_breadth_v2(leader_metrics, expected_leaders=SLPM_EXPECTED_LEADERS):
    # N1 ronda 2 (2026-10-14, dictamen auditor, C1-C10):
    #   - Fail-closed por componente: si falta el campo requerido en
    #     cualquier lider, ese componente queda NaN.
    #   - composite = NaN si CUALQUIER componente es NaN
    #     (INSUFFICIENT_RS_RATIO o INSUFFICIENT_COMPONENT_DATA).
    #   - INPUT_UNAVAILABLE para None, NO_LEADERS para lista vacia.
    #   - Valores no finitos (+Inf/-Inf/strings/OverflowError)
    #     tratados como ausentes.
    #   - No hay fallback a rs nominal (P1.1).
    if leader_metrics is None:
        return _breadth_unavailable("INPUT_UNAVAILABLE", expected_leaders)
    if len(leader_metrics) == 0:
        return _breadth_unavailable("NO_LEADERS", expected_leaders)

    n = len(leader_metrics)
    coverage = min(n / expected_leaders, 1.0) if expected_leaders > 0 else 0.0

    def _rs_positive(m):
        if not m:
            return None
        vf = _safe_float(m.get("rs_ratio_20d"))
        return None if vf is None else vf > 0.0

    def _momentum_positive(m):
        if not m:
            return None
        # A3.3-06: fallback rs_momentum -> rs_mom_20.
        # Si faltan ambos, cuenta como ausente (no como 0).
        raw = m.get("rs_momentum")
        if raw is None:
            raw = m.get("rs_mom_20")
        vf = _safe_float(raw)
        return None if vf is None else vf > 0.0

    def _flow_positive(m):
        if not m:
            return None
        vf = _safe_float(m.get("flow_proxy_z"))
        return None if vf is None else vf > 0.0

    def _wyckoff_favorable(m):
        if not m:
            return None
        phase = m.get("wyckoff_phase")
        if phase not in _VALID_WYCKOFF_PHASES:
            return None
        return phase in ("ACCUMULATION", "MARKUP")

    def _component(leaders, fn):
        """(value, missing_count). value = NaN si missing > 0."""
        vals = [fn(m) for m in leaders]
        valid = [v for v in vals if v is not None]
        missing = len(vals) - len(valid)
        if missing > 0:
            return float("nan"), missing
        return sum(1 for v in valid if v) / len(valid), 0

    rs_positive, rs_missing = _component(leader_metrics, _rs_positive)
    momentum_positive, mom_missing = _component(leader_metrics, _momentum_positive)
    flow_positive, flow_missing = _component(leader_metrics, _flow_positive)
    wyckoff_favorable, wyck_missing = _component(leader_metrics, _wyckoff_favorable)

    rs_breadth_status = "OK" if rs_missing == 0 else "INSUFFICIENT_RS_RATIO"

    all_ok = (
        rs_missing == 0 and mom_missing == 0
        and flow_missing == 0 and wyck_missing == 0
    )

    if all_ok:
        composite = (
            SLPM_WEIGHTS["leader_breadth"]["rs"] * rs_positive
            + SLPM_WEIGHTS["leader_breadth"]["momentum"] * momentum_positive
            + SLPM_WEIGHTS["leader_breadth"]["flow"] * flow_positive
            + SLPM_WEIGHTS["leader_breadth"]["wyckoff"] * wyckoff_favorable
        )
        effective_composite = (
            composite * coverage if coverage < 0.5 else composite
        )
        leader_breadth_status = "OK"
    else:
        composite = float("nan")
        effective_composite = float("nan")
        if rs_missing > 0:
            leader_breadth_status = "INSUFFICIENT_RS_RATIO"
        else:
            leader_breadth_status = "INSUFFICIENT_COMPONENT_DATA"

    return {
        "rs_breadth": rs_positive,
        "rs_breadth_status": rs_breadth_status,
        "rs_breadth_missing": rs_missing,
        "leader_breadth_status": leader_breadth_status,
        "momentum_breadth": momentum_positive,
        "momentum_breadth_missing": mom_missing,
        "flow_breadth": flow_positive,
        "flow_breadth_missing": flow_missing,
        "wyckoff_breadth": wyckoff_favorable,
        "wyckoff_breadth_missing": wyck_missing,
        "composite": composite,
        "effective_composite": effective_composite,
        "n_used": n,
        "expected_leaders": expected_leaders,
        "coverage": coverage,
        "coverage_warning": coverage < 0.5,
    }

def compute_leader_integrity(leader_metrics):
    # N2 (2026-10-13, auditor): LIS usa `rs` (ratio de precio
    # nominal accion/ETF) como input. El resto del sistema migro a
    # `rs_ratio_20d` (retorno relativo). Cambiar la escala de input
    # de LIS saturaria tanh y desplazaria la confidence en produccion.
    # Requiere recalibracion estadistica previa. LIS queda PROVISIONAL
    # y NO validado como senal decisoria. Ver TRIAGE del auditor
    # 2026-10-13, hallazgo N2.
    if not leader_metrics:
        return {'lis': 0.0, 'n_leaders': 0, 'lis_validated': False}
    # P1.1 (2026-10-13): WONT FIX en este fix. El LIS sigue usando
    # `rs` (precio/ETF) como input. Cambiar a `rs_ratio_20d` (retorno
    # relativo, escala +-0.10 tipica vs +-1.5 del ratio de precios)
    # saturaria tanh(-) y desplazaria la confidence en produccion
    # para todos los runs. Requiere recalibracion estadistica previa
    # (mismo caso que F2.4-13: no tocar sin evidencia). Documentado
    # como deuda viva en el TRIAGE del auditor 2026-10-13.
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
        # Fix 2026-09-29: 'or np.nan' convertia flow_proxy_z=0.0 en NaN.
        # 0.0 es un valor legitimo (flujo neutro); su ausencia es la que
        # debe ir a NaN. Mismo patron que la familia de invariantes de
        # finitud ya cerrada esta sesion (S-03c, PeriodState.sshprnamt,
        # extract_sshprnamt_by_figi, _validate_with_cache, flow_pct_assets).
        flow = m.get('flow_proxy_z')
        if flow is None:
            flow = np.nan
        flow_norm = np.tanh(flow / 2)
        wyckoff_map = {'MARKUP': 1.0, 'ACCUMULATION': 0.75, 'RANGE': 0.0, 'DISTRIBUTION': -0.75, 'MARKDOWN': -1.0}
        wyckoff_score = wyckoff_map.get(m.get('wyckoff_phase', ''), 0.0)
        individual = SLPM_WEIGHTS["lis"]["rs"] * rs_norm + SLPM_WEIGHTS["lis"]["momentum"] * mom_norm + SLPM_WEIGHTS["lis"]["flow"] * flow_norm + SLPM_WEIGHTS["lis"]["wyckoff"] * wyckoff_score
        scores.append(individual)
    if not scores:
        return {'lis': 0.0, 'n_leaders': 0, 'lis_validated': False}
    lis_val = _safe_mean(scores)
    if np.isnan(lis_val):
        lis_val = 0.0
    return {'lis': float(np.clip(lis_val, -1, 1)), 'n_leaders': len(scores),
            'lis_validated': False}  # N2: provisional

def compute_flow_divergence_v2(leader_metrics, sector_flow_proxy_z, sector_price_flow=None):
    sector_flow_proxy_z = sector_flow_proxy_z if sector_flow_proxy_z is not None else np.nan
    leader_flows = [m.get('flow_proxy_z', np.nan) for m in (leader_metrics or []) if m and 'flow_proxy_z' in m]
    valid_leader_flows = [f for f in leader_flows if pd.notna(f)]
    if not valid_leader_flows:
        leader_flow_div = np.nan
        structural_flow_div = np.nan
    else:
        leader_flow_div = float(_safe_mean(valid_leader_flows) - sector_flow_proxy_z)
        leader_flow_std = safe_std(valid_leader_flows) if len(valid_leader_flows) > 1 else np.nan
        structural_flow_div = float(_safe_mean(valid_leader_flows) - leader_flow_std) if pd.notna(leader_flow_std) else np.nan
    sector_flow_vs_price_div = float(sector_flow_proxy_z - sector_price_flow) if (sector_price_flow is not None and pd.notna(sector_price_flow)) else np.nan
    composite = 0.50 * leader_flow_div + 0.25 * sector_flow_vs_price_div + 0.25 * structural_flow_div
    return {'leader_flow_div': leader_flow_div, 'sector_flow_vs_price_div': sector_flow_vs_price_div, 'structural_flow_div': structural_flow_div, 'composite': composite}

def evaluate_slpm_v12(df_market, sector_results, leader_metrics, top_sector_flow,
                       tactical_scores=None, structural_scores=None, sector_persistence=None,
                       temporal_meta=None):
    ranking = sector_results.get('ranking', [])
    if not ranking:
        return {'sector': '', 'sector_etf': '', 'state': 'UNRESOLVED', 'opportunity_quadrant': 'Transition'}
    sector_etf = ranking[0][0]
    sector_name = SECTOR_NAMES.get(sector_etf, sector_etf)

    breadth_v2 = compute_leader_breadth_v2(leader_metrics, expected_leaders=SLPM_EXPECTED_LEADERS)
    integrity = compute_leader_integrity(leader_metrics)
    top_sector_flow = top_sector_flow if top_sector_flow is not None else np.nan
    flow_div_v2 = compute_flow_divergence_v2(leader_metrics, top_sector_flow, None)

    tactical_score = tactical_scores.get(sector_etf, 0.0) if tactical_scores else 0.0
    structural_score = structural_scores.get(sector_etf, 0.0) if structural_scores else 0.0
    persistence = sector_persistence.get(sector_etf) if sector_persistence else None
    if persistence is None:
        persistence = 0.5
        print(f"    SLPM v1.2: Persistence no disponible para {sector_name}, usando 0.5 neutro.")

    effective_breadth = breadth_v2['effective_composite']
    lb_status = breadth_v2.get('leader_breadth_status', 'UNKNOWN')

    # N1 ronda 2 (2026-10-14, dictamen auditor, opcion D commit 3):
    #   cortocircuito principal. Si el breadth no esta validado, no se
    #   llama a classify_leadership_state con un composite no finito.
    #   La causa ya viene identificada en breadth_v2 y se propaga
    #   como reason_code en el dict final.
    if lb_status != 'OK':
        if lb_status == 'INPUT_UNAVAILABLE':
            instant_reason = 'Breadth de lideres no disponible (entrada no disponible).'
            _sp = ranking[0][3] if len(ranking[0]) > 3 else 'desconocida'
            instant_reason += f" El sector lider ({sector_name}) esta en fase {_sp}."
            instant_reason_code = 'INPUT_UNAVAILABLE'
            data_quality = 'LOW'
        elif lb_status == 'NO_LEADERS':
            instant_reason = 'Breadth de lideres no disponible (sin lideres evaluables).'
            _sp = ranking[0][3] if len(ranking[0]) > 3 else 'desconocida'
            instant_reason += f" El sector lider ({sector_name}) esta en fase {_sp}."
            instant_reason_code = 'NO_LEADERS'
            data_quality = 'LOW'
        elif lb_status == 'INSUFFICIENT_RS_RATIO':
            _n = breadth_v2.get('rs_breadth_missing', 0)
            instant_reason = f"Breadth de lideres no disponible (rs_ratio_20d ausente en {_n} lideres)."
            instant_reason_code = 'INSUFFICIENT_DATA_RS'
            data_quality = 'LOW'
        elif lb_status == 'INSUFFICIENT_COMPONENT_DATA':
            instant_reason = 'Breadth de lideres no disponible (componentes de breadth incompletos).'
            instant_reason_code = 'INSUFFICIENT_COMPONENT_DATA'
            data_quality = 'LOW'
        else:
            instant_reason = f"Breadth de lideres no disponible (status={lb_status})."
            instant_reason_code = 'INSUFFICIENT_DATA_BREADTH'
            data_quality = 'LOW'
        instant_state = 'UNRESOLVED'
    else:
        result = classify_leadership_state(structural_score, effective_breadth, persistence, coverage=breadth_v2['coverage'])
        instant_state = result['state']
        instant_reason = result['reason']
        instant_reason_code = result.get('reason_code', 'UNKNOWN')
        data_quality = result.get('data_quality', 'UNKNOWN')
        if not leader_metrics:
            sector_phase = ranking[0][3] if len(ranking[0]) > 3 else 'desconocida'
            instant_reason += f" El sector lider ({sector_name}) esta en fase {sector_phase}, lo que impide calcular metricas de lideres."

    transition_data = confirm_transition(sector_etf, instant_state)
    state = transition_data['confirmed_state']
    reason = instant_reason
    reason_code = instant_reason_code
    quadrant = get_opportunity_quadrant(state)

    errors = validate_state(state, structural_score, effective_breadth, persistence)
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
        'data_quality': data_quality,
        'narrow_leadership': (
            flow_div_v2.get('leader_flow_div', 0) > 0.2 and
            top_sector_flow < 0.2 and
            len(leader_metrics) >= 3
        )
    }
