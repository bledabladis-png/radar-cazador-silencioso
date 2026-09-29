import pandas as pd
from indicators.volatility import volatility_regime


def compute_volatility_regime(returns):
    """Devuelve (z_series, regime, confidence).

    C-08 (2026-09-29): regime = "N/D" si el ultimo z es NaN o la serie
    esta vacia. Antes caia silenciosamente a "STRESS" (todas las
    comparaciones con NaN dan False, caia al else). Bug observado:
    1803/1884 detecciones historicas de STRESS eran falsos positivos
    por NaN heredado de huecos de VIX (90 huecos en 10 anios x 20 dias
    de rolling std x 756 dias de baseline).
    """
    z = volatility_regime(returns)
    if z.empty:
        return z, "N/D", 0.0
    last = z.iloc[-1]
    if pd.isna(last):
        return z, "N/D", 0.0

    if last < -0.5:
        regime = 'LOW'
    elif last < 0.5:
        regime = 'NORMAL'
    elif last < 1.5:
        regime = 'ELEVATED'
    else:
        regime = 'STRESS'

    confidence = min(abs(last) / 2, 1.0)
    return z, regime, confidence
