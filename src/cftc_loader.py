import pandas as pd
import numpy as np
import os
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import CFTC_MARKETS

def load_cftc_manual(path="data/cftc_raw.txt"):
    if not os.path.exists(path):
        print(f"[CFTC] Archivo no encontrado: {path}")
        return None
    try:
        df = pd.read_csv(path, header=None, low_memory=False, dtype=str)
        print(f"[CFTC] Archivo cargado: {len(df)} filas")
        return df
    except Exception as e:
        print(f"[CFTC] Error al cargar: {e}")
        return None

def parse_cftc_financials(df_raw):
    if df_raw is None or len(df_raw) == 0:
        return None
    try:
        parsed = pd.DataFrame()
        parsed['market'] = df_raw[0].str.strip()
        parsed['date'] = pd.to_datetime(df_raw[2], format='%Y-%m-%d', errors='coerce')
        parsed['asset_long'] = pd.to_numeric(df_raw[8], errors='coerce')
        parsed['asset_short'] = pd.to_numeric(df_raw[9], errors='coerce')
        parsed = parsed.dropna(subset=['date', 'asset_long', 'asset_short'])
        if parsed.empty:
            return None
        parsed['net_position'] = parsed['asset_long'] - parsed['asset_short']
        return parsed
    except Exception as e:
        print(f"[CFTC] Error en parseo: {e}")
        return None

def compute_cftc_signal(df, window=52):
    df = df.copy().sort_values('date')
    df['mean'] = df['net_position'].rolling(window, min_periods=10).mean()
    df['std'] = df['net_position'].rolling(window, min_periods=10).std()
    df['cftc_z'] = (df['net_position'] - df['mean']) / df['std']
    return df

def get_latest_cftc_signal_for_market(df_parsed, market_substring):
    """Retorna la fila más reciente (última fecha) para un mercado que contiene la subcadena"""
    mask = df_parsed['market'].str.contains(market_substring, case=False, na=False)
    df_market = df_parsed[mask].copy()
    if df_market.empty:
        return None
    df_market = df_market.sort_values('date')
    return df_market.iloc[-1].to_dict()
def compute_cftc_signal_enhanced(df):
    net = df['asset_manager_net']
    delta = net.diff()
    zscore = (net - net.rolling(52, min_periods=10).mean()) / net.rolling(52, min_periods=10).std()
    delta_z = (delta - delta.rolling(52, min_periods=10).mean()) / delta.rolling(52, min_periods=10).std()
    return pd.DataFrame({
        'cftc_z': zscore,
        'cftc_delta_z': delta_z,
        'net': net,
        'delta': delta
    })
