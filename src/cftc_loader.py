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

def update_cftc_history(raw_file="data/cftc_raw.txt", history_file="data/cftc_history.csv"):
    """
    Actualiza el histórico CFTC con los datos del archivo semanal raw.
    Si el histórico no existe, lo crea.
    Elimina duplicados por fecha y mercado.
    """
    # Cargar histórico existente
    if os.path.exists(history_file):
        hist = pd.read_csv(history_file)
        hist['date'] = pd.to_datetime(hist['date'])
    else:
        hist = pd.DataFrame()
    
    # Cargar nuevo raw
    if not os.path.exists(raw_file):
        print(f"[CFTC] No se encontró {raw_file}. No se actualiza histórico.")
        return hist
    
    df_raw = pd.read_csv(raw_file, header=None)
    parsed = parse_cftc_financials(df_raw)
    if parsed is None or parsed.empty:
        print("[CFTC] No se pudieron parsear los datos nuevos.")
        return hist
    
    parsed['date'] = pd.to_datetime(parsed['date'])
    
    # Combinar y eliminar duplicados (por fecha y mercado)
    combined = pd.concat([hist, parsed], ignore_index=True)
    combined = combined.drop_duplicates(subset=['date', 'market'])
    combined = combined.sort_values('date')
    
    # Guardar histórico
    combined.to_csv(history_file, index=False)
    print(f"[CFTC] Histórico actualizado: {len(combined)} registros totales.")
    return combined

def get_cftc_history(history_file="data/cftc_history.csv"):
    """Carga el histórico CFTC si existe."""
    if os.path.exists(history_file):
        return pd.read_csv(history_file)
    return None

def compute_cftc_zscore_from_history(history_file="data/cftc_history.csv", window=52):
    """
    Calcula el z-score de CFTC para cada mercado usando el histórico acumulado.
    Ventana de 52 semanas (por defecto).
    """
    if not os.path.exists(history_file):
        print("[CFTC] Histórico no encontrado. Ejecute update_cftc_history primero.")
        return None
    
    hist = pd.read_csv(history_file)
    hist['date'] = pd.to_datetime(hist['date'])
    result = []
    
    for market in hist['market'].unique():
        df_m = hist[hist['market'] == market].copy().sort_values('date')
        if len(df_m) < 10:  # Mínimo para calcular rolling
            continue
        df_m['mean'] = df_m['net_position'].rolling(window, min_periods=10).mean()
        df_m['std'] = df_m['net_position'].rolling(window, min_periods=10).std()
        df_m['cftc_z'] = (df_m['net_position'] - df_m['mean']) / df_m['std']
        result.append(df_m)
    
    if result:
        return pd.concat(result, ignore_index=True)
    return None