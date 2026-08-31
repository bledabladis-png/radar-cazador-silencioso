# -*- coding: utf-8 -*-
"""
regimes/financial_conditions.py -- Financial Conditions Score (v3.15 corregido)

Evalua el estres de las condiciones financieras mediante VIX, credito (HYG/LQD),
dolar (DXY) y curva de tipos (TNX-FVX).

CORRECCIONES APLICADAS (auditoria 24/07/2026):
  - Signo de HYG/LQD: -tanh -> +tanh (HYG/LQD alto = apetito por riesgo = menos estres).
  - Ventana de curva: 120 -> 60 (consistente con el resto de modulos).
  - Peso de DXY: 0.20 -> 0.15 (sobreponderacion corregida).
  - Nombre de funcion: compute_liquidity_score -> compute_financial_conditions.
"""
import pandas as pd
import numpy as np
from src.utils import robust_zscore, get_col
from config.settings import FINANCIAL_CONDITIONS_WEIGHTS, FINANCIAL_CONDITIONS_THRESHOLDS

def compute_financial_conditions(df):
    """Calcula el score de condiciones financieras agregando VIX, credito, dolar y curva."""
    scores = pd.DataFrame(index=df.index)

    # VIX (invertido: a mas VIX, mas estres)
    try:
        vix = get_col(df, '^VIX', 'Close')
        scores['vix'] = -np.tanh(robust_zscore(vix, 60))
    except KeyError:
        pass

    # Credito HYG/LQD (CORREGIDO: +tanh, HYG/LQD alto = menos estres)
    try:
        hyg = get_col(df, 'HYG', 'Close')
        lqd = get_col(df, 'LQD', 'Close')
        ratio = hyg / lqd
        scores['credit'] = np.tanh(robust_zscore(ratio, 60))
    except KeyError:
        pass

    # Dolar (invertido: dolar fuerte = estres, peso reducido a 0.15)
    try:
        dxy = get_col(df, 'DX-Y.NYB', 'Close')
        scores['dollar'] = -np.tanh(robust_zscore(dxy.pct_change(fill_method=None), 60))
    except KeyError:
        pass

    # Curva 10Y-2Y (ventana unificada a 60)
    try:
        tnx = get_col(df, '^TNX', 'Close')
        fvx = get_col(df, '^FVX', 'Close')
        curve = tnx - fvx
        scores['curve'] = np.tanh(robust_zscore(curve, 60))
    except KeyError:
        pass

    weights = FINANCIAL_CONDITIONS_WEIGHTS
    available = [c for c in weights if c in scores.columns]
    w_sum = sum(weights[c] for c in available)
    if w_sum == 0:
        return pd.Series(0, index=df.index), 'NEUTRAL', 1.0

    financial_score = sum(scores[c] * weights[c] / w_sum for c in available)
    confidence = (1 - scores[available].std(axis=1).fillna(0) / 2).clip(0, 1)
    last = financial_score.iloc[-1] if not financial_score.empty else 0

    if last > FINANCIAL_CONDITIONS_THRESHOLDS['abundante']:
        regime = 'ABUNDANTE'
    elif last > FINANCIAL_CONDITIONS_THRESHOLDS['neutral']:
        regime = 'NEUTRAL'
    elif last > FINANCIAL_CONDITIONS_THRESHOLDS['estrecha']:
        regime = 'ESTRECHA'
    elif last > FINANCIAL_CONDITIONS_THRESHOLDS['high_stress']:
        regime = 'HIGH_STRESS'
    else:
        regime = 'EXTREME_STRESS'

    return financial_score, regime, confidence.iloc[-1] if not confidence.empty else 0.5


# Mantener compatibilidad con codigo existente que importa compute_liquidity_score
def compute_liquidity_score(df):
    return compute_financial_conditions(df)

compute_financial_conditions.__doc__ = f"""
Formula: {FINANCIAL_CONDITIONS_WEIGHTS['vix']:.2f}*VIX_norm + {FINANCIAL_CONDITIONS_WEIGHTS['credit']:.2f}*Credito_norm + {FINANCIAL_CONDITIONS_WEIGHTS['dollar']:.2f}*Dolar_norm + {FINANCIAL_CONDITIONS_WEIGHTS['curve']:.2f}*Curva_norm. Cada componente se normaliza con z-score robusto y tanh. Retorna puntuacion [-1, +1].
"""
