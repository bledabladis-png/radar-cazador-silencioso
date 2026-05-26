import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__)))
from features import compute_features, robust_zscore
from config import tickers

# ---------- Funciones de precio (sin cambios) ----------
def compute_sector_momentum(df, sectors, benchmark='SPY', window=20):
    momentum = {}
    for sec in sectors:
        rel = df[sec] / df[benchmark]
        mom = rel.pct_change(periods=window, fill_method=None).iloc[-1]
        momentum[sec] = mom
    return momentum

def rank_sectors(momentum_dict):
    return sorted(momentum_dict.items(), key=lambda x: x[1], reverse=True)

def interpret_regime(breadth_signal, dispersion, stress, vix_z):
    if stress == 1 or vix_z > 1.5:
        return "RISK OFF (estres alto)", "Reducir exposicion a ciclicos, aumentar defensivos o efectivo."
    if breadth_signal > 0.2 and dispersion < 0.02:
        return "TREND STRONG (seguir lider)", "Mantener exposicion en el sector lider."
    if breadth_signal < -0.2 and dispersion > 0.03:
        return "ROTATION ACTIVE (cambio de lider)", "Rotacion activa. Esperar nuevo lider."
    if breadth_signal > 0:
        return "EXPANSION (rotacion moderada)", "Entorno favorable para ciclicos. Seguir rotacion."
    return "NEUTRAL", "Sin senal clara. Esperar."

def run_radar(df):
    features = compute_features(df)
    sectors = tickers['sectors']
    momentum = compute_sector_momentum(df, sectors)
    ranking = rank_sectors(momentum)
    dispersion = features['sector_dispersion'].iloc[-1]
    breadth = features['breadth_signal'].iloc[-1]
    stress = features['stress'].iloc[-1]
    vix_z = features['vix_z'].iloc[-1]
    regime, accion = interpret_regime(breadth, dispersion, stress, vix_z)
    return ranking, dispersion, breadth, vix_z, stress, regime, accion

# ---------- RADAR DE FLUJOS (v3.17, histórico completo) ----------
def run_flow_radar(df, sectors=None):
    if sectors is None:
        sectors = tickers['sectors']

    flow_mom = pd.DataFrame(index=df.index)

    for sec in sectors:
        close = df[sec]
        volume = df.get(f"{sec}_volume")
        if volume is None:
            flow_mom[sec] = 0.0
            continue
        dollar_vol = close * volume
        ret = close.pct_change(fill_method=None)
        flow = ret * dollar_vol
        # Z-score robusto con ventana 60
        flow_z = robust_zscore(flow, window=60)
        # Suavizado EWMA (span=10) para el ranking
        flow_mom[sec] = flow_z.ewm(span=10, min_periods=20).mean()

    # Última fila para ranking
    latest_flow = flow_mom.iloc[-1]
    ranking_flow = latest_flow.sort_values(ascending=False)

    flow_dispersion = flow_mom.std(axis=1).iloc[-1]
    flow_breadth = (latest_flow > 0).mean()

    if flow_breadth > 0.6:
        regime_flow = "ENTRADA GENERALIZADA (riesgo-on)"
    elif flow_dispersion > 0.5:
        regime_flow = "ROTACION FUERTE (flujos dispersos)"
    else:
        regime_flow = "FLUJO SELECTIVO"

    return ranking_flow, flow_dispersion, flow_breadth, regime_flow, flow_mom