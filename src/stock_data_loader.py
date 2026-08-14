import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import time

CACHE_HOURS = 23

YAHOO_TICKER_MAP = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
    "MOGA": "MOG-A",
}

def normalize_yahoo_ticker(t):
    """Convierte tickers problemáticos al formato que acepta Yahoo Finance."""
    return YAHOO_TICKER_MAP.get(t, t)

def get_stock_list():
    """Obtiene lista de tickers para descargar: top 20 de cada sector y de cada índice."""
    tickers = []

    # 1) Sectores USA (data/etf_holdings.csv)
    try:
        df_sect = pd.read_csv('data/etf_holdings.csv')
        if 'weight' in df_sect.columns:
            df_sect = df_sect.sort_values(['etf', 'weight'], ascending=[True, False])
        for etf, group in df_sect.groupby('etf'):
            tickers.extend([normalize_yahoo_ticker(t) for t in group['ticker'].head(20).tolist()])
    except Exception:
        pass

    # 2) Índices americanos/europeos (data/index_holdings.csv)
    try:
        df_idx = pd.read_csv('data/index_holdings.csv')
        if 'weight' in df_idx.columns:
            df_idx = df_idx.sort_values(['etf', 'weight'], ascending=[True, False])
        for etf, group in df_idx.groupby('etf'):
            tickers.extend([normalize_yahoo_ticker(t) for t in group['ticker'].head(20).tolist()])
    except Exception:
        pass

    # Eliminar duplicados preservando orden
    seen = set()
    result = []
    for t in tickers:
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result

def _get_yf_session():
    try:
        from curl_cffi import requests as curl_requests
        return curl_requests.Session(impersonate="chrome")
    except Exception:
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_stock_prices():
    cache_path = 'data/stock_prices.csv'
    if os.path.exists(cache_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - mtime < timedelta(hours=CACHE_HOURS):
            return pd.read_csv(cache_path, header=[0,1], index_col=0, parse_dates=True)

    tickers = get_stock_list()
    if not tickers:
        return None

    batch_size = 5
    delay = 2
    all_data = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"Descargando lote {i//batch_size + 1}: {batch}")
        try:
            session = _get_yf_session()
            if session:
                data_batch = yf.download(batch, period='5y', auto_adjust=True, session=session)
            else:
                data_batch = yf.download(batch, period='5y', auto_adjust=True)
            if not data_batch.empty:
                all_data.append(data_batch)
        except Exception as e:
            print(f"Error en lote {batch}: {e}")
        if i + batch_size < len(tickers):
            time.sleep(delay)

    if not all_data:
        return None

    data = pd.concat(all_data, axis=1)
    if not isinstance(data.columns, pd.MultiIndex):
        data.columns = pd.MultiIndex.from_tuples(data.columns)

    data.to_csv(cache_path)
    return data
