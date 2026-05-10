# global_risk_v4.py – Risk Module (régimen de volatilidad y dirección)
# Mide dirección del precio, volatilidad y breadth de riesgo.
# Independiente del Flow Module.

import pandas as pd
import numpy as np

def compute_risk_direction(weekly_returns, window=20):
    """
    Risk Direction: retorno semanal dividido por volatilidad.
    Similar a la "direction" del PWM antiguo, pero sin volumen.
    """
    direction_df = pd.DataFrame(index=weekly_returns.index)
    
    for t in weekly_returns.columns:
        ret = weekly_returns[t]
        vol = ret.rolling(window).std()
        vol_rolling_quantile = vol.rolling(252, min_periods=20).quantile(0.1).iloc[-1]
        vol_floor = max(vol_rolling_quantile, 0.01) if pd.notna(vol_rolling_quantile) else 0.01
        adj_vol = vol.clip(lower=vol_floor)
        direction_df[t] = ret / adj_vol
    
    return direction_df

def compute_volatility_regime(weekly_returns, assets, window=20):
    """
    Volatility Regime: percentil de la volatilidad actual vs histórico 3 años.
    0-1 = baja volatilidad, 1-2 = normal, >2 = alto estrés.
    """
    regimes = {}
    
    for t in assets:
        if t not in weekly_returns.columns:
            continue
        ret = weekly_returns[t].dropna()
        if len(ret) < window:        # <--- añade esta línea
            regimes[t] = 1.0          # <--- añade esta línea
            continue                  # <--- añade esta línea
        if len(ret) < 156:  # 3 años mínimo
            regimes[t] = 1.0
            continue
        
        vol_current = ret.iloc[-window:].std()
        vol_median = ret.rolling(156).std().median()
        vol_mad = (ret.rolling(156).std() - ret.rolling(156).std().median()).abs().median()
        
        z_vol = (vol_current - vol_median) / (1.4826 * vol_mad + 1e-9)
        regimes[t] = z_vol
    
    # Agregado: mediana de los z-scores
    if regimes:
        return np.median(list(regimes.values()))
    return 1.0

def compute_risk_breadth(direction_df, assets):
    """
    Risk Breadth: % de activos con dirección positiva.
    Independiente del volumen.
    """
    latest = direction_df.iloc[-1]
    pos_count = sum(1 for t in assets if latest.get(t, 0) > 0)
    return pos_count / len(assets) if assets else 0