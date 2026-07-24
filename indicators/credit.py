# -*- coding: utf-8 -*-
"""
indicators/credit.py -- Credit Risk Appetite Signal (v3.15 corregido)
Mide el apetito relativo por riesgo crediticio mediante dos componentes:
  1. HYG/LQD: High Yield vs Investment Grade (apetito por credito de alto rendimiento)
  2. LQD/IEF: Investment Grade Corporates vs Intermediate Treasuries (apetito por credito frente a deuda publica)

Ambos componentes se normalizan con z-score robusto (60d) y tanh.
La senhal final es la media ponderada 60/40.

Orientacion:
  +1 = condiciones crediticias favorables (risk-on)
  -1 = estres crediticio (risk-off)

NOTA: Este modulo NO calcula un "credit spread" en sentido estricto (OAS, CDS, yield spread).
Calcula ratios de fortaleza relativa entre ETFs de credito.
"""
import pandas as pd
import numpy as np
from src.utils import get_col, robust_zscore

def credit_risk_signal(df_market):
    """
    Calcula la senhal de apetito por riesgo crediticio.
    
    Componentes:
      - HYG/LQD: High Yield vs Investment Grade relative strength.
      - LQD/IEF: Investment Grade Corporates vs Intermediate Treasuries relative strength.
    
    Retorna una Series con la senhal normalizada en [-1, 1].
    """
    try:
        hyg = get_col(df_market, 'HYG', 'Close')
        lqd = get_col(df_market, 'LQD', 'Close')
        ief = get_col(df_market, 'IEF', 'Close')
    except KeyError:
        return pd.Series(dtype=float)

    # Alinear datos y eliminar NaN
    data = pd.concat([hyg.rename('HYG'), lqd.rename('LQD'), ief.rename('IEF')], axis=1)
    data = data.dropna()

    if len(data) < 60:
        return pd.Series(index=data.index, dtype=float)

    # Ratios de precio relativo
    ratio_hyg_lqd = data['HYG'] / data['LQD']
    ratio_lqd_ief = data['LQD'] / data['IEF']

    # Z-scores robustos
    z_hyg_lqd = robust_zscore(ratio_hyg_lqd, window=60)
    z_lqd_ief = robust_zscore(ratio_lqd_ief, window=60)

    # Componentes normalizados
    credit_component = np.tanh(z_hyg_lqd)       # HYG vs LQD
    treasury_component = np.tanh(z_lqd_ief)     # LQD vs IEF

    # Senhal compuesta
    credit = 0.60 * credit_component + 0.40 * treasury_component
    credit.name = 'credit_signal'

    return credit


# Mantener compatibilidad con codigo existente que importa credit_spread_signal
def credit_spread_signal(df_market):
    """
    DEPRECATED: Usar credit_risk_signal() en su lugar.
    Mantenido por compatibilidad con modulos existentes.
    """
    return credit_risk_signal(df_market)
