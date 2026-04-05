import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from features import compute_features, compute_etf_flows_robust as compute_etf_flows
from features import compute_flow_momentum, compute_flow_dispersion, compute_flow_breadth
from config import tickers

# ---------- Funciones existentes (precio) ----------
def compute_sector_momentum(df, sectors, benchmark='SPY', window=20):
    momentum = {}
    for sec in sectors:
        rel = df[sec] / df[benchmark]
        mom = rel.pct_change(window).iloc[-1]
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

# ---------- RADAR DE FLUJOS ----------
def run_flow_radar(df, sectors=None):
    if sectors is None:
        sectors = tickers['sectors']
    flows = compute_etf_flows(df, sectors)
    flow_mom = compute_flow_momentum(flows)
    # Suavizado adicional para alinear horizontes (ventana 3 dias)
    flow_mom = flow_mom.rolling(3).mean()
    latest_flow = flow_mom.iloc[-1]
    ranking_flow = latest_flow.sort_values(ascending=False)
    flow_dispersion = compute_flow_dispersion(flow_mom).iloc[-1]
    flow_breadth = compute_flow_breadth(flow_mom).iloc[-1]
    if flow_breadth > 0.6:
        regime_flow = "ENTRADA GENERALIZADA (riesgo-on)"
    elif flow_dispersion > 0.5:
        regime_flow = "ROTACION FUERTE (flujos dispersos)"
    else:
        regime_flow = "FLUJO SELECTIVO"
    return ranking_flow, flow_dispersion, flow_breadth, regime_flow, flow_mom

# ---------- Funciones auxiliares ----------
def compute_flow_acceleration(flow_mom, window=5):
    return flow_mom - flow_mom.rolling(window).mean()

def compute_volume_zscore(df, sectors, window=20):
    vol_z = pd.DataFrame(index=df.index)
    for sec in sectors:
        dollar_vol = df[f"{sec}_dollar_vol_smoothed"]
        mean = dollar_vol.rolling(window).mean()
        std = dollar_vol.rolling(window).std()
        vol_z[sec] = (dollar_vol - mean) / std
    return vol_z

# ---------- Funciones de distribucion ----------
def divergence_score(price_mom, flow_mom):
    p = np.tanh(price_mom)
    f = np.tanh(flow_mom)
    return p * (-f)

def distribution_prob_continuous(price_mom, flow_mom, flow_acc, vol_z):
    p = np.tanh(price_mom)
    f = np.tanh(flow_mom)
    a = np.tanh(flow_acc)
    v = np.tanh(vol_z)
    x = 0.4 * (p * (-f)) + 0.3 * (-a) + 0.2 * (-f) + 0.1 * v
    prob = 1 / (1 + np.exp(-6 * (x - 0.2)))
    return prob

def distribution_score_binary(price_mom, flow_mom, flow_acc, vol_z, weights=None):
    if weights is None:
        weights = {'divergence': 0.4, 'flow_neg': 0.2, 'flow_acc': 0.3, 'volume': 0.1}
    score = 0.0
    if price_mom > 0 and flow_mom < 0:
        score += weights['divergence']
    if flow_mom < -0.1:
        score += weights['flow_neg']
    if flow_acc < 0:
        score += weights['flow_acc']
    if vol_z > 1:
        score += weights['volume']
    return score

def prob_distribution_binary(score, k=5):
    return 1 / (1 + np.exp(-k * (score - 0.5)))
