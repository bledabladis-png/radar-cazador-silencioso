"""
global_data_loader.py – Descarga y caché de datos para el Radar Global v3.19.
No genera señales de trading; solo proporciona OHLCV para el módulo global.
"""

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime, timedelta
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from global_config_v4 import FLOW_ASSETS, RISK_ASSETS, CROSS_REGION

CACHE_FILE = "data/global_market_data.csv"

# Mapeo Stooq para activos globales
STOOQ_MAPPING = {
    'SPY': 'spy.us',
    'EZU': 'ezu.us',
    'EWJ': 'ewj.us',
    'EEM': 'eem.us',
    'VGK': 'vgk.us',
    'IWM': 'iwm.us',
    'FXI': 'fxi.us',
    'TLT': 'tlt.us',
    'HYG': 'hyg.us',
    'LQD': 'lqd.us',
    'GLD': 'gld.us',
    'DBC': 'dbc.us',
    'XOP': 'xop.us',
    'UUP': 'uup.us',
    'ACWI': 'acwi.us',
    'EURUSD=X': 'eurusd',
    'JPYUSD=X': 'jpyusd'
}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def download_global_batch(tickers_list, start, end):
    """Descarga un lote de tickers globales con reintentos."""
    return yf.download(
        tickers_list,
        start=start,
        end=end,
        interval='1d',
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by='ticker'
    )

def fetch_stooq_global(ticker, start, end):
    """Fallback a Stooq para un ticker global individual."""
    import requests
    from io import StringIO

    symbol = STOOQ_MAPPING.get(ticker, ticker.lower().replace("^", "") + ".us")
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        if df.empty:
            return None
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        df = df.loc[start:end]
        if df.empty:
            return None
        result = pd.DataFrame({
            f"{ticker}_close": df['Close'],
            f"{ticker}_volume": df['Volume'],
            f"{ticker}_open": df['Open'],
            f"{ticker}_high": df['High'],
            f"{ticker}_low": df['Low']
        })
        return result
    except Exception as e:
        print(f"[GlobalStooq] Error con {ticker}: {e}")
        return None

def download_global_market_data(force=False):
    """
    Descarga datos OHLCV de los activos globales con caché de 23 horas.
    Retorna DataFrame con columnas: TICKER_close, TICKER_volume, etc.
    """
    if not force and os.path.exists(CACHE_FILE):
        file_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        if datetime.now() - file_time < timedelta(hours=23):
            print(f"[GlobalDataLoader] Usando caché de datos globales ({CACHE_FILE}).")
            return pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)

    # Recolectar todos los tickers desde la configuración v4.0
    all_global = (
        FLOW_ASSETS['equity'] +
        FLOW_ASSETS['fixed_income'] +
        FLOW_ASSETS['commodities'] +
        [FLOW_ASSETS['dollar_proxy']] +
        ['EURUSD=X', 'JPYUSD=X'] +
        ['ACWI'] +
        ['VGK', 'IWM', 'FXI', 'LQD']  # Contexto extendido
    )
    # Eliminar duplicados manteniendo el orden
    all_global = list(dict.fromkeys(all_global))

    end = datetime.now()
    start = end - timedelta(days=3650)

    all_data = pd.DataFrame()

    # Descarga principal desde Yahoo
    try:
        data = download_global_batch(all_global, start, end)
        if data is not None and not data.empty:
            for ticker in all_global:
                if isinstance(data.columns, pd.MultiIndex) and ticker in data.columns.levels[0]:
                    df_ticker = data[ticker]
                    temp = pd.DataFrame({
                        f"{ticker}_close": df_ticker['Close'],
                        f"{ticker}_volume": df_ticker['Volume'],
                        f"{ticker}_open": df_ticker['Open'],
                        f"{ticker}_high": df_ticker['High'],
                        f"{ticker}_low": df_ticker['Low']
                    })
                    all_data = pd.concat([all_data, temp], axis=1)
            print("[GlobalDataLoader] Datos globales descargados desde Yahoo Finance.")
    except Exception as e:
        print(f"[GlobalDataLoader] Yahoo falló para globales: {e}")

    # Fallback Stooq para los tickers que faltan
    missing = [t for t in all_global if f"{t}_close" not in all_data.columns]
    if missing:
        print(f"[GlobalDataLoader] Descargando {len(missing)} tickers faltantes desde Stooq...")
        for t in missing:
            df_stooq = fetch_stooq_global(t, start, end)
            if df_stooq is not None and not df_stooq.empty:
                all_data = pd.concat([all_data, df_stooq], axis=1)

    if all_data.empty:
        raise Exception("No se pudieron descargar datos globales.")

    all_data.to_csv(CACHE_FILE)
    print(f"[GlobalDataLoader] Caché global guardada en {CACHE_FILE}")
    return all_data

if __name__ == '__main__':
    download_global_market_data()