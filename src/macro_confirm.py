"""
macro_confirm.py – Capa Macro simplificada (solo score continuo)
Versión 3.0 – Causal Institutional Model
No genera señales de trading; solo información.
"""

import numpy as np
import pandas as pd
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from features import robust_zscore

def compute_macro_score(df, breadth_signal=0.0):
    """
    Calcula un score de régimen macro continuo entre -1 y +1.
    Combina crédito (HYG/LQD), curva (10Y-2Y), VIX (nivel + tendencia) y breadth.
    
    Parámetros:
    - df: DataFrame con columnas 'HYG', 'LQD', '^TNX', '^IRX', '^VIX'
    - breadth_signal: valor numérico (opcional, por defecto 0). Se espera que sea un número entre -1 y 1.
    
    Retorna:
    - macro_score: float entre -1 y +1 (negativo = riesgo-off, positivo = riesgo-on)
    """
    # 1. Crédito (HYG/LQD) -> señal en [-1,1]
    credit_ratio = df['HYG'] / df['LQD']
    credit_z = robust_zscore(credit_ratio, window=60).iloc[-1]
    credit_norm = 1 - np.clip(credit_z, 0, 2) / 2          # [0,1]
    credit_signal = 2 * (credit_norm - 0.5)               # [-1,1]

    # 2. Curva de tipos (10Y - 2Y)
    yield_10y = df['^TNX'] / 100.0
    yield_2y = df['^IRX'] / 100.0
    curve = yield_10y - yield_2y
    curve_z = robust_zscore(curve, window=120).iloc[-1]
    curve_norm = 1 - np.clip(curve_z, 0, 2) / 2           # [0,1]
    curve_signal = 2 * (curve_norm - 0.5)                 # [-1,1]

    # 3. VIX (nivel + tendencia)
    vix_series = df['^VIX']
    vix_z = robust_zscore(vix_series, window=60).iloc[-1]
    # Tendencia de los últimos 5 días (pendiente)
    if len(vix_series) >= 5:
        slope = np.polyfit(range(5), vix_series.tail(5), 1)[0]
        vix_trend = np.tanh(slope)                       # [-1,1]
    else:
        vix_trend = 0.0
    vix_component = np.exp(-vix_z) * (1 - vix_trend)     # (0,1]
    vix_signal = 2 * (vix_component - 0.5)               # [-1,1]

    # 4. Breadth (opcional, se puede pasar desde run.py)
    # Se espera que breadth_signal esté normalizado aproximadamente en [-1,1]
    breadth_signal_clipped = np.clip(breadth_signal, -1, 1)

    # 5. Combinación (pesos: 0.3 crédito, 0.3 curva, 0.2 VIX, 0.2 breadth)
    raw = (0.3 * credit_signal +
           0.3 * curve_signal +
           0.2 * vix_signal +
           0.2 * breadth_signal_clipped)
    macro_score = np.tanh(raw)                           # [-1, +1]
    return macro_score

# (Opcional) Función de formateo para el reporte Markdown
def format_macro_section(macro_score):
    lines = []
    lines.append("\n## Contexto Macro (Causal)\n")
    lines.append(f"**Macro Score:** {macro_score:.2f} (continuo, -1 = RISK-OFF, +1 = RISK-ON)\n")
    return lines