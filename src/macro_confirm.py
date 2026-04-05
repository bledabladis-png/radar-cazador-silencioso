"""
macro_confirm.py – Capa 4 de confirmación macro institucional
Versión 2.1 (con escalas normalizadas, agreement tanh, confidence y failsafe)
No modifica el core del radar. Solo añade contexto interpretativo.
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from features import robust_zscore

# ------------------------------------------------------------
# Componentes individuales (normalizados) – devuelven escalares (último valor)
# ------------------------------------------------------------

def compute_credit_score(df, hyg_ticker='HYG', lqd_ticker='LQD'):
    """
    Riesgo sistémico basado en ratio HYG/LQD.
    Retorna score compuesto (0.7*z + 0.3*mom), además de z y mom individuales (escalares).
    """
    ratio = df[hyg_ticker] / df[lqd_ticker]
    z = robust_zscore(ratio, window=60).iloc[-1]          # último valor
    mom_raw = ratio.pct_change(10)
    mom = robust_zscore(mom_raw, window=20).iloc[-1]
    score = 0.7 * z + 0.3 * mom
    return score, z, mom

def compute_curve_score(df, ten_y_ticker='^TNX', two_y_ticker='^IRX'):
    """
    Curva de tipos (10Y - 2Y). Normaliza nivel y dinámica.
    """
    yield_10y = df[ten_y_ticker] / 100.0
    yield_2y = df[two_y_ticker] / 100.0
    curve = yield_10y - yield_2y
    z = robust_zscore(curve, window=120).iloc[-1]
    mom_raw = curve.diff(20)
    mom = robust_zscore(mom_raw, window=60).iloc[-1]
    score = 0.6 * z + 0.4 * mom
    return score, z, mom

def compute_sentiment_score(df, spy_ticker='SPY', vix_ticker='^VIX', vxz_ticker='VXZ', vxx_ticker='VXX'):
    """
    Sentimiento de mercado: SPY/VIX + estructura temporal de volatilidad (VXZ/VXX) si está disponible.
    """
    spy_vix = df[spy_ticker] / df[vix_ticker]
    spy_vix_z = robust_zscore(spy_vix, window=60).iloc[-1]
    # Si existen VXZ y VXX, usamos estructura temporal; si no, solo SPY/VIX
    if vxz_ticker in df.columns and vxx_ticker in df.columns:
        vix_term = df[vxz_ticker] / df[vxx_ticker]
        vix_term_z = robust_zscore(vix_term, window=60).iloc[-1]
        score = 0.6 * spy_vix_z + 0.4 * vix_term_z
        return score, spy_vix_z, vix_term_z
    else:
        # Fallback: solo SPY/VIX
        return spy_vix_z, spy_vix_z, None

# ------------------------------------------------------------
# Flujo de caja para obtener signo del régimen de flujo del radar
# ------------------------------------------------------------

def get_flow_regime_sign(flow_breadth):
    """
    Convierte el breadth de flujo en signo: +1 (entrada generalizada), -1 (salidas), 0 (neutral).
    Umbrales: >0.55 -> positivo, <0.45 -> negativo.
    """
    if flow_breadth > 0.55:
        return 1
    elif flow_breadth < 0.45:
        return -1
    else:
        return 0

# ------------------------------------------------------------
# Función principal que agrega todo y devuelve diccionario con resultados escalares
# ------------------------------------------------------------

def compute_macro_confirm(df, flow_regime_sign):
    """
    Calcula el score macro, consistencia (agreement), confianza y alineación.
    Retorna un diccionario listo para ser volcado al reporte.
    """
    credit_score, credit_z, credit_mom = compute_credit_score(df)
    curve_score, curve_z, curve_mom = compute_curve_score(df)
    sentiment_score, spy_vix_z, vix_term_z = compute_sentiment_score(df)

    signals = np.array([credit_score, curve_score, sentiment_score])
    # Agreement suavizado con tanh (intensidad)
    agreement = np.mean(np.tanh(signals))
    # Confianza = magnitud media de las señales (sin tanh)
    confidence = np.mean(np.abs(signals))

    # Score bruto (ponderación)
    raw_score = 0.4 * credit_score + 0.35 * curve_score + 0.25 * sentiment_score
    # Ajuste por consistencia
    macro_score = raw_score * (1 + 0.25 * agreement)
    macro_score = np.clip(macro_score, -2, 2)

    # Estado de confianza (para failsafe)
    if confidence < 0.2:
        confidence_state = "INCIERTO"
    elif confidence < 0.6:
        confidence_state = "MODERADO"
    else:
        confidence_state = "ALTO"

    # Régimen macro cualitativo
    if macro_score > 1.0:
        macro_regime = "RISK-ON FUERTE"
    elif macro_score > 0:
        macro_regime = "EXPANSION"
    elif macro_score > -1.0:
        macro_regime = "TRANSICION"
    else:
        macro_regime = "RISK-OFF"

    # Alineación con el flujo del radar
    flow_sign = flow_regime_sign
    if flow_sign == 0:
        alignment = "NEUTRAL (flujo sin dirección)"
    elif np.sign(macro_score) == flow_sign:
        alignment = "ALINEADO"
    else:
        alignment = "DIVERGENTE"

    # Advertencia especial si hay divergencia fuerte
    warning = ""
    if alignment == "DIVERGENTE" and confidence > 0.5:
        warning = "⚠️ Divergencia fuerte macro-flujo"

    # Diccionario de salida
    return {
        'macro_score': macro_score,
        'macro_regime': macro_regime,
        'confidence': confidence,
        'confidence_state': confidence_state,
        'alignment': alignment,
        'warning': warning,
        'components': {
            'credit': credit_score,
            'curve': curve_score,
            'sentiment': sentiment_score
        },
        'raw_details': {
            'credit_z': credit_z,
            'credit_mom': credit_mom,
            'curve_z': curve_z,
            'curve_mom': curve_mom,
            'spy_vix_z': spy_vix_z,
            'vix_term_z': vix_term_z
        }
    }

def format_macro_section(macro_data):
    """
    Genera líneas Markdown para la sección de confirmación macro.
    """
    lines = []
    lines.append("\n## Confirmación Macro Institucional\n")
    lines.append(f"**Macro Score:** {macro_data['macro_score']:.2f}\n")
    lines.append(f"**Régimen:** {macro_data['macro_regime']}\n")
    lines.append(f"**Confianza:** {macro_data['confidence']:.2f} ({macro_data['confidence_state']})\n")
    lines.append(f"**Alineación con flujo sectorial:** {macro_data['alignment']}\n")
    if macro_data['warning']:
        lines.append(f"**⚠️ {macro_data['warning']}**\n")
    
    # Desglose de componentes (opcional pero útil)
    comp = macro_data['components']
    lines.append("\n**Componentes:**\n")
    lines.append(f"- Crédito: {comp['credit']:.2f}\n")
    lines.append(f"- Curva: {comp['curve']:.2f}\n")
    lines.append(f"- Sentimiento: {comp['sentiment']:.2f}\n")
    
    return lines