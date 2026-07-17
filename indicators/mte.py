# -*- coding: utf-8 -*-
"""
mte.py -- Market Transition Engine v1.0
Motor de inferencia macroeconómica basado en flujos institucionales.
"""
import pandas as pd
import numpy as np
from src.utils import robust_zscore, get_col
import json
import os

# ============================================================
# 0. FUNCIONES AUXILIARES
# ============================================================
def tanh(x):
    return np.tanh(x)

def _get_last(x):
    """Extrae el último valor de una Series o float."""
    if isinstance(x, pd.Series):
        return x.iloc[-1]
    return x

# ============================================================
# 1. SECTOR ROTATION SCORE (SRS)
# ============================================================
def sector_rotation_score(df_market):
    cyclical = ['XLK', 'XLY', 'XLI', 'XLF', 'XLB', 'XLE']
    defensive = ['XLU', 'XLP', 'XLV', 'XLRE', 'XLC']
    sectors = cyclical + defensive

    rs = {}
    for s in sectors:
        try:
            close_s = get_col(df_market, s, 'Close')
            close_spy = get_col(df_market, '^GSPC', 'Close')
            rs[s] = close_s / close_spy
        except KeyError:
            continue

    if not rs:
        return 0.0

    rs_cyclical = pd.concat([rs[s] for s in cyclical if s in rs], axis=1).mean(axis=1)
    rs_defensive = pd.concat([rs[s] for s in defensive if s in rs], axis=1).mean(axis=1)

    spread = rs_defensive - rs_cyclical
    mom_spread = spread.pct_change(20)
    speed_spread = (spread - spread.shift(20)).abs()

    # Dispersión cross‑sectional
    rs_all = pd.concat([rs[s] for s in sectors if s in rs], axis=1)
    dispersion = rs_all.std(axis=1)

    # Amplitud interna: % de sectores defensivos que baten al SPY
    try:
        spy_close = get_col(df_market, '^GSPC', 'Close')
        spy_mom = spy_close.pct_change(20)
        defensive_mom = pd.concat([rs[s].pct_change(20) for s in defensive if s in rs], axis=1)
        defensive_beating_spy = (defensive_mom.gt(spy_mom, axis=0)).mean(axis=1)
    except:
        defensive_beating_spy = pd.Series(0.5, index=df_market.index)

    z_spread   = robust_zscore(spread, 60)
    z_mom      = robust_zscore(mom_spread, 60)
    z_speed    = robust_zscore(speed_spread, 60)
    z_disp     = robust_zscore(dispersion, 60)
    z_breadth_def = robust_zscore(defensive_beating_spy, 60)

    srs = (0.35 * tanh(z_spread) +
           0.20 * tanh(z_mom) +
           0.15 * tanh(z_speed) +
           0.15 * tanh(z_disp) +
           0.15 * tanh(z_breadth_def))
    return _get_last(srs)


# ============================================================
# 2. SAFE HAVEN SCORE (SHS)
# ============================================================
def safe_haven_score(df_market):
    hard_havens = ['GLD', 'SLV', 'TLT']
    defensive_havens = ['XLP', 'XLV', 'XLU', 'QUAL', 'IEF', 'BIL']

    def haven_subscore(tickers):
        signals = []
        for t in tickers:
            try:
                close = get_col(df_market, t, 'Close')
                mom = close.pct_change(20)
                signals.append(tanh(robust_zscore(mom, 60)))
            except KeyError:
                pass
        if not signals:
            return 0.0
        return _get_last(pd.concat(signals, axis=1).mean(axis=1))

    hard_score = haven_subscore(hard_havens)
    defensive_score = haven_subscore(defensive_havens)
    return 0.6 * hard_score + 0.4 * defensive_score


# ============================================================
# 3. CREDIT STRESS SCORE (CLS)
# ============================================================
def credit_stress_score(financial_conditions, credit_signal,
                        volatility_signal, vix_term, darkpool_z, pcr_z):
    def stress(val):
        if val is None or not pd.notna(val):
            return 0.5
        return float(np.clip(np.tanh(val / 2), 0, 1))

    fc_stress = stress(-financial_conditions)
    dp_stress = stress(darkpool_z)
    liquidity_family = np.mean([fc_stress, dp_stress])

    credit_stress_val = stress(-credit_signal)
    credit_family = credit_stress_val

    vol_stress = stress(volatility_signal)
    vix_stress = stress(vix_term)
    volatility_family = np.mean([vol_stress, vix_stress])

    pcr_stress = stress(pcr_z)
    sentiment_family = pcr_stress

    return float(np.mean([liquidity_family, credit_family, volatility_family, sentiment_family]))


# ============================================================
# 4. INFLATION PRESSURE SCORE (IPS)
# ============================================================
def inflation_pressure_score(df_market):
    assets = ['XLE', '^SPGSCI', 'TIP']
    signals = []
    for t in assets:
        try:
            close = get_col(df_market, t, 'Close')
            mom = close.pct_change(20)
            signals.append(tanh(robust_zscore(mom, 60)))
        except KeyError:
            pass

    try:
        tip_close = get_col(df_market, 'TIP', 'Close')
        ief_close = get_col(df_market, 'IEF', 'Close')
        tip_ief_ratio = tip_close / ief_close
        tip_ief_mom = tip_ief_ratio.pct_change(20)
        signals.append(tanh(robust_zscore(tip_ief_mom, 60)))
    except KeyError:
        pass

    if not signals:
        return 0.0

    ips_raw = pd.concat(signals, axis=1).mean(axis=1)
    ips = ips_raw.ewm(span=20).mean()
    return _get_last(ips)


# ============================================================
# 5. ÍNDICES AGREGADOS
# ============================================================
def compute_msi(srs, shs, cls):
    srs_mapped = (srs + 1) / 2
    shs_mapped = (shs + 1) / 2
    raw = 0.40 * srs_mapped + 0.35 * cls + 0.25 * shs_mapped
    return max(0, min(100, raw * 100))

def compute_ipi(ips):
    ips_mapped = (ips + 1) / 2
    return max(0, min(100, ips_mapped * 100))


# ============================================================
# 6. SISTEMA DE PUNTUACIÓN DE ESCENARIOS
# ============================================================
SCENARIO_WEIGHTS = {
    "CLS": {"weight": 3, "reason": "El deterioro financiero es condición necesaria en una crisis."},
    "SHS": {"weight": 2, "reason": "Los activos refugio suelen liderar durante las fases defensivas."},
    "SRS": {"weight": 2, "reason": "La rotación sectorial suele preceder al deterioro macro."},
    "IPS": {"weight": 2, "reason": "La inflación diferencia recesión de estanflación."}
}

def score_scenarios(srs, shs, cls, ips):
    scores = {}

    # CRISIS (prioridad máxima: bonus base por cumplir condiciones)
    crisis = 0
    if cls > 0.5 and shs > 0.3 and srs > 0.3:
        crisis += 6  # Base por cumplir las 3 condiciones simultáneamente
    if cls > 0.5: crisis += SCENARIO_WEIGHTS["CLS"]["weight"]
    if shs > 0.3: crisis += SCENARIO_WEIGHTS["SHS"]["weight"]
    if srs > 0.3: crisis += SCENARIO_WEIGHTS["SRS"]["weight"]
    if cls > 0.7: crisis += 2
    if cls > 0.85: crisis += 3
    scores['CRISIS'] = crisis

    # RECESSION
    recession = 0
    if cls > 0.2: recession += SCENARIO_WEIGHTS["CLS"]["weight"]
    if shs > 0.2: recession += SCENARIO_WEIGHTS["SHS"]["weight"]
    if srs > 0.2: recession += SCENARIO_WEIGHTS["SRS"]["weight"]
    if cls > 0.4: recession += 1
    scores['RECESSION'] = recession

    # STAGFLATION
    stagflation = 0
    if ips > 0.3: stagflation += SCENARIO_WEIGHTS["IPS"]["weight"]
    if srs > 0: stagflation += SCENARIO_WEIGHTS["SRS"]["weight"]
    if cls < 0.3: stagflation += SCENARIO_WEIGHTS["CLS"]["weight"]
    if ips > 0.5: stagflation += 2
    scores['STAGFLATION'] = stagflation

    # SOFT LANDING
    soft = 0
    if srs > 0.1: soft += SCENARIO_WEIGHTS["SRS"]["weight"]
    if shs > 0.1: soft += SCENARIO_WEIGHTS["SHS"]["weight"]
    if cls < 0: soft += SCENARIO_WEIGHTS["CLS"]["weight"]
    if cls < -0.2: soft += 1
    scores['SOFT LANDING'] = soft

    # EXPANSION
    expansion = 0
    if srs < -0.1: expansion += SCENARIO_WEIGHTS["SRS"]["weight"]
    if cls < -0.1: expansion += SCENARIO_WEIGHTS["CLS"]["weight"]
    if shs < 0: expansion += SCENARIO_WEIGHTS["SHS"]["weight"]
    if srs < -0.3: expansion += 1
    scores['EXPANSION'] = expansion

    scores['MIXED'] = 3
    return scores


# ============================================================
# 7. MATRIZ DE TRANSICIONES
# ============================================================
NORMAL_TRANSITIONS = {
    'EXPANSION':      ['SOFT LANDING', 'MIXED'],
    'SOFT LANDING':   ['EXPANSION', 'RECESSION', 'MIXED'],
    'RECESSION':      ['SOFT LANDING', 'CRISIS', 'STAGFLATION', 'MIXED'],
    'STAGFLATION':    ['RECESSION', 'MIXED'],
    'CRISIS':         ['RECESSION', 'MIXED'],
    'MIXED':          ['EXPANSION', 'SOFT LANDING', 'RECESSION', 'STAGFLATION', 'CRISIS'],
}

EXCEPTION_TRANSITIONS = {
    ('EXPANSION', 'RECESSION'): "Salto abrupto por estrés extremo",
    ('EXPANSION', 'CRISIS'):    "Evento de mercado excepcional (ej. COVID)",
    ('SOFT LANDING', 'CRISIS'): "Deterioro súbito de condiciones financieras",
}

def validate_transition(previous, current, cls):
    if current in NORMAL_TRANSITIONS.get(previous, []):
        return True
    if (previous, current) in EXCEPTION_TRANSITIONS and cls > 0.85:
        return True
    return False


# ============================================================
# 8. PERSISTENCIA DE ESTADO (HISTÉRESIS)
# ============================================================
STATE_FILE = 'outputs/mte_state.json'

def load_previous_scenario():
    try:
        with open(STATE_FILE, 'r') as f:
            data = json.load(f)
            return data.get('scenario', 'MIXED'), data.get('pending', None)
    except:
        return 'MIXED', None

def save_scenario(scenario, pending=None):
    os.makedirs('outputs', exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump({'scenario': scenario, 'pending': pending}, f)


# ============================================================
# 9. CONFIANZA
# ============================================================
def consensus_score(srs, shs, cls, ips):
    values = np.array([srs, shs, cls, ips])
    median = np.median(values)
    distance = np.abs(values - median)
    mean_distance = distance.mean()
    return float(1 - np.clip(mean_distance / 2.0, 0, 1))

def distance_to_threshold(srs, shs, cls, ips, scenario):
    """Distancia media a los umbrales del escenario, normalizada a [0,1]."""
    if scenario == 'CRISIS':
        distances = [cls - 0.5, shs - 0.3, srs - 0.3]
    elif scenario == 'RECESSION':
        distances = [cls - 0.2, shs - 0.2, srs - 0.2]
    elif scenario == 'STAGFLATION':
        distances = [ips - 0.3, srs, 0.3 - cls]
    elif scenario == 'SOFT LANDING':
        distances = [srs - 0.1, shs - 0.1, -cls]
    elif scenario == 'EXPANSION':
        distances = [-srs - 0.1, -cls - 0.1, -shs]
    else:
        return 0.5
    return float(np.clip(np.mean([max(0, d) for d in distances]) / 0.5, 0, 1))

def compute_confidence(srs, shs, cls, ips, scenario):
    distance_conf = distance_to_threshold(srs, shs, cls, ips, scenario)
    consensus_conf = consensus_score(srs, shs, cls, ips)
    return float(np.clip(0.6 * distance_conf + 0.4 * consensus_conf, 0, 1))


# ============================================================
# 10. CLASIFICADOR PRINCIPAL
# ============================================================
def classify_mte(srs, shs, cls, ips):
    scores = score_scenarios(srs, shs, cls, ips)
    new_scenario = max(scores, key=scores.get)

    # Desempate
    if list(scores.values()).count(scores[new_scenario]) > 1:
        priority = ['CRISIS', 'RECESSION', 'STAGFLATION', 'SOFT LANDING', 'EXPANSION', 'MIXED']
        for s in priority:
            if scores[s] == scores[new_scenario]:
                new_scenario = s
                break

    # Validar transición
    prev_scenario, pending = load_previous_scenario()
    if not validate_transition(prev_scenario, new_scenario, cls):
        new_scenario = prev_scenario

    # Histéresis adaptativa
    if cls > 0.85:
        save_scenario(new_scenario)
        final_scenario = new_scenario
    elif new_scenario != prev_scenario:
        if pending is None:
            save_scenario(prev_scenario, new_scenario)
            final_scenario = prev_scenario
        elif pending == new_scenario:
            save_scenario(new_scenario)
            final_scenario = new_scenario
        else:
            save_scenario(prev_scenario, None)
            final_scenario = prev_scenario
    else:
        if pending is not None:
            save_scenario(new_scenario)
        final_scenario = new_scenario

    confidence = compute_confidence(srs, shs, cls, ips, final_scenario)
    return final_scenario, confidence


# ============================================================
# 11. FUNCIÓN PRINCIPAL (ORQUESTADOR)
# ============================================================
def compute_mte(df_market, financial_conditions_score, credit_signal,
                volatility_signal, pcr_data=None, darkpool_data=None):
    """
    Calcula el MTE completo y devuelve un diccionario con todos los resultados.
    """
    try:
        # Extraer scores existentes
        fc = _get_last(financial_conditions_score)
        cred = _get_last(credit_signal)
        vol = _get_last(volatility_signal)

        # VIX term (VIX3M - VIX)
        try:
            vix_close = get_col(df_market, '^VIX', 'Close')
            vix3m_close = get_col(df_market, '^VIX3M', 'Close')
            vix_term = _get_last(tanh(robust_zscore(vix3m_close - vix_close, 60)))
        except:
            vix_term = 0.0

        # Dark Pool Z-Score
        darkpool_z = darkpool_data.get('z_score', None) if darkpool_data else None

        # PCR Z-Score
        pcr_z = pcr_data.get('z_score', None) if pcr_data else None

        # Calcular los 4 motores
        srs = sector_rotation_score(df_market)
        shs = safe_haven_score(df_market)
        cls = credit_stress_score(fc, cred, vol, vix_term, darkpool_z, pcr_z)
        ips = inflation_pressure_score(df_market)

        # Índices
        msi = compute_msi(srs, shs, cls)
        ipi = compute_ipi(ips)

        # Escenario
        scenario, confidence = classify_mte(srs, shs, cls, ips)

        return {
            'scenario': scenario,
            'confidence': confidence,
            'msi': msi,
            'ipi': ipi,
            'srs': srs,
            'shs': shs,
            'cls': cls,
            'ips': ips
        }
    except Exception as e:
        print(f"  MTE: Error - {e}")
        return None
