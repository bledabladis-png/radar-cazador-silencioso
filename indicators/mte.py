# -*- coding: utf-8 -*-
"""
mte.py -- Market Transition Engine v1.0
Motor de inferencia macroecon�mica basado en flujos institucionales.
"""
import pandas as pd
import numpy as np
import json
import os
from config.settings import MTE_STATE_FILE
from src.utils import robust_zscore, get_col
import json
import os

# ============================================================
# 0. FUNCIONES AUXILIARES
# ============================================================
def tanh(x):
    return np.tanh(x)

def _get_last(x):
    """Extrae el �ltimo valor de una Series o float."""
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

    # Dispersi�n cross-sectional
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
                        volatility_signal, vix_term, darkpool_z, pcr_z,
                        nfci_series=None, credit_oas_series=None):
    """
    CLS v1.1 - Arquitectura por familias con NFCI y Credit OAS.
    Si Credit OAS no est� disponible, usa HYG/LQD como proxy.
    Si NFCI no est� disponible, usa financial_conditions como proxy.
    Bloquea CRISIS si alguna familia es NaN (retorna NaN).
    """
    def robust_zscore_series(series, window=104):
        if len(series) < 20:
            return pd.Series([0.0], index=series.index)
        median = series.rolling(window, min_periods=20).median()
        def mad_func(x):
            return np.median(np.abs(x - np.median(x)))
        mad = series.rolling(window, min_periods=20).apply(mad_func, raw=True)
        return (series - median) / (1.4826 * mad + 1e-9)
    
    def stress_transform(z):
        return float(np.clip(np.tanh(z.iloc[-1] / 2.0), 0, 1)) if len(z) > 0 else 0.5
    
    def stress(val):
        if val is None or not pd.notna(val):
            return 0.5
        return float(np.clip(np.tanh(val / 2), 0, 1))
    
    # Familia 1: Liquidez (NFCI si existe, si no Financial Conditions)
    # Verificar si NFCI tiene datos V�LIDOS para la fecha actual
    nfci_valid = False
    if nfci_series is not None and len(nfci_series) > 0:
        nfci_val = nfci_series.iloc[-1] if hasattr(nfci_series, 'iloc') else nfci_series
        if pd.notna(nfci_val) and np.isfinite(nfci_val):
            nfci_valid = True
    
    if nfci_valid:
        nfci_z = robust_zscore_series(nfci_series)
        nfci_stress = stress_transform(nfci_z)
    else:
        nfci_stress = stress(-financial_conditions) if financial_conditions is not None else 0.5
    
    # Familia 2: Cr�dito (Credit OAS si existe, si no HYG/LQD)
    # Verificar si Credit OAS tiene datos V�LIDOS para la fecha actual
    oas_valid = False
    if credit_oas_series is not None and len(credit_oas_series) > 0:
        oas_val = credit_oas_series.iloc[-1] if hasattr(credit_oas_series, 'iloc') else credit_oas_series
        if pd.notna(oas_val) and np.isfinite(oas_val):
            oas_valid = True
    
    if oas_valid:
        oas_z = robust_zscore_series(credit_oas_series)
        oas_stress = stress_transform(oas_z)
        hyg_stress = stress(-credit_signal) if credit_signal is not None else 0.5
        credit_stress_val = 0.60 * oas_stress + 0.40 * hyg_stress
    else:
        # Fallback a HYG/LQD cuando Credit OAS no est� disponible
        credit_stress_val = stress(-credit_signal) if credit_signal is not None else 0.5
    
    # Familia 3: Volatilidad (VIX)
    vix_stress = stress(volatility_signal) if volatility_signal is not None else 0.5
    
    # Familia 4: Complementarios
    pcr_stress = stress(pcr_z) if pcr_z is not None else 0.5
    dp_stress = stress(darkpool_z) if (darkpool_z is not None and not np.isnan(darkpool_z)) else 0.5
    complementary_stress = 0.50 * pcr_stress + 0.50 * dp_stress
    
    cls = (0.25 * nfci_stress +
           0.35 * credit_stress_val +
           0.25 * vix_stress +
           0.15 * complementary_stress)
    
    # Bloquear si alg�n componente es NaN
    if np.isnan(cls):
        return np.nan
    
    return float(np.clip(cls, 0.0, 1.0))


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
# 5. �NDICES AGREGADOS
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
# 6. SISTEMA DE PUNTUACI�N DE ESCENARIOS
# ============================================================
SCENARIO_WEIGHTS = {
    "CLS": {"weight": 3, "reason": "El deterioro financiero es condici�n necesaria en una crisis."},
    "SHS": {"weight": 2, "reason": "Los activos refugio suelen liderar durante las fases defensivas."},
    "SRS": {"weight": 2, "reason": "La rotaci�n sectorial suele preceder al deterioro macro."},
    "IPS": {"weight": 2, "reason": "La inflaci�n diferencia recesi�n de estanflaci�n."}
}

def score_scenarios(srs, shs, cls, ips):
    scores = {}

    # CRISIS (prioridad m�xima: bonus base por cumplir condiciones)
    crisis = 0
    if not np.isfinite(cls):
        crisis = -999  # CLS inv�lido ? CRISIS bloqueado
    else:
        if cls > 0.5:
            crisis += 6  # CLS > 0.5 ya indica estr�s financiero extremo
        if cls > 0.7: crisis += 3  # Bonus adicional por estr�s severo
        if shs > 0.3: crisis += SCENARIO_WEIGHTS["SHS"]["weight"]
        if srs > 0.3: crisis += SCENARIO_WEIGHTS["SRS"]["weight"]
    if cls > 0.7: crisis += 2
    if cls > 0.85: crisis += 3
    scores['CRISIS'] = crisis

    # RECESSION
    recession = 0
    if np.isfinite(cls):
        if cls > 0.25 and srs > 0.1 and cls <= 0.5: recession += 1  # bonus base
        if cls > 0.2: recession += SCENARIO_WEIGHTS["CLS"]["weight"]
        if cls > 0.4 and cls <= 0.5: recession += 1
    if shs > 0.2: recession += SCENARIO_WEIGHTS["SHS"]["weight"]
    if srs > 0.2: recession += SCENARIO_WEIGHTS["SRS"]["weight"]
    scores['RECESSION'] = recession

    # STAGFLATION
    stagflation = 0
    if ips > 0.15 and srs > 0: stagflation += 1  # bonus base (umbral reducido)
    if ips > 0.15: stagflation += 3  # IPS es el factor distintivo de STAGFLATION (umbral reducido)
    if srs > 0: stagflation += SCENARIO_WEIGHTS["SRS"]["weight"]
    if cls < 0.3: stagflation += SCENARIO_WEIGHTS["CLS"]["weight"]
    if ips > 0.5: stagflation += 2
    scores['STAGFLATION'] = stagflation

    # SOFT LANDING
    soft = 0
    if srs > 0.1 and shs > 0.1: soft += 1  # bonus base
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

    scores['MIXED'] = 2
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
    ('EXPANSION', 'RECESSION'): "Salto abrupto por estr�s extremo",
    ('EXPANSION', 'CRISIS'):    "Evento de mercado excepcional (ej. COVID)",
    ('SOFT LANDING', 'CRISIS'): "Deterioro s�bito de condiciones financieras",
}

def validate_transition(previous, current, cls):
    if current in NORMAL_TRANSITIONS.get(previous, []):
        return True
    if (previous, current) in EXCEPTION_TRANSITIONS and cls > 0.85:
        return True
    return False


# ============================================================
# 8. PERSISTENCIA DE ESTADO (HIST�RESIS)
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

    # Validar transici�n
    prev_scenario, pending = load_previous_scenario()
    if not validate_transition(prev_scenario, new_scenario, cls):
        new_scenario = prev_scenario

    # Hist�resis adaptativa
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
    if confidence == 0.0:
        print(f"    MTE: Confianza 0% en escenario {final_scenario}. Forzando MIXED.")
        final_scenario = 'MIXED'
    return final_scenario, confidence


# ============================================================
# 11. FUNCI�N PRINCIPAL (ORQUESTADOR)
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

        # �ndices
        msi = compute_msi(srs, shs, cls)
        ipi = compute_ipi(ips)

        # Escenario
        scenario, confidence = classify_mte(srs, shs, cls, ips)

        # Guardar estado en JSON para trazabilidad
        try:
            os.makedirs(os.path.dirname(MTE_STATE_FILE), exist_ok=True)
            with open(MTE_STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    'scenario': scenario,
                    'confidence': confidence,
                    'msi': msi,
                    'ipi': ipi,
                    'srs': srs,
                    'shs': shs,
                    'cls': cls,
                    'ips': ips
                }, f, indent=2, default=str)
        except Exception as e:
            print(f'  MTE: No se pudo guardar estado JSON - {e}')

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


