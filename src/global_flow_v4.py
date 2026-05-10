# global_flow_v4.py – Flow Module (presión de capital)
# Mide presión de volumen monetario por activo y por bloque.
# No utiliza dirección de precio (eso va en el Risk Module).

import pandas as pd
import numpy as np
from features import robust_zscore

def compute_flow_pressure(weekly_returns, dollar_vol, window=60):
    """
    Flow Pressure: z-score robusto del dollar volume semanal.
    No incorpora dirección del precio.
    Retorna DataFrame con flow_pressure por activo.
    """
    flow_df = pd.DataFrame(index=weekly_returns.index)
    
    for t in weekly_returns.columns:
        if t not in dollar_vol.columns:
            continue
        dv = dollar_vol[t].reindex(weekly_returns.index)
        # Z-score del dollar volume (presión de capital pura)
        flow_z = robust_zscore(dv, window=window)
        # Suavizado EWMA
        flow_df[t] = flow_z.ewm(halflife=2, min_periods=10).mean()
    
    return flow_df

def compute_capital_rotation(flow_df, equity_assets, fixed_income_assets, commodities_assets):
    """
    Capital Rotation: diferencia entre flujo a RV y flujo a refugio.
    Positivo = capital rotando hacia riesgo.
    Negativo = capital rotando hacia seguridad.
    """
    latest = flow_df.iloc[-1]
    
    equity_flow = np.median([latest.get(t, 0) for t in equity_assets if t in latest.index])
    fi_flow = np.median([latest.get(t, 0) for t in fixed_income_assets if t in latest.index])
    commodity_flow = np.median([latest.get(t, 0) for t in commodities_assets if t in latest.index])
    
    risk_flow = (equity_flow + commodity_flow) / 2.0
    safe_flow = fi_flow
    
    rotation = risk_flow - safe_flow
    return rotation

def flow_participation_ratio(flow_df, equity_assets):
    """
    Participation Ratio: % de activos de RV con presión de flujo positiva.
    """
    latest = flow_df.iloc[-1]
    pos_count = sum(1 for t in equity_assets if latest.get(t, 0) > 0)
    return pos_count / len(equity_assets) if equity_assets else 0