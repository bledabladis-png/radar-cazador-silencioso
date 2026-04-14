import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys, os
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import tickers

# =========================================================
# FUNCIONES DE DESCARGA INDIVIDUAL (FALLBACK)
# =========================================================

def fetch_stooq(ticker, start, end):
    """
    Descarga datos OHLCV desde Stooq para un ticker.
    Retorna DataFrame con columnas: open, high, low, close, volume.
    """
    mapping = {
        'SPY': 'spy.us',
        'XLK': 'xlk.us',
        'XLF': 'xlf.us',
        'XLE': 'xle.us',
        'XLI': 'xli.us',
        'XLY': 'xly.us',
        'XLP': 'xlp.us',
        'XLV': 'xlv.us',
        'XLU': 'xlu.us',
        'XLRE': 'xlre.us',
        'QQQ': 'qqq.us',
        'ACWI': 'acwi.us',
        'HYG': 'hyg.us',
        'LQD': 'lqd.us',
        'TLT': 'tlt.us',
        '^VIX': 'vix.us',
        '^TNX': '10usy.b',
        '^IRX': '2usy.b',
    }
    symbol = mapping.get(ticker, ticker.lower().replace("^", "") + ".us")
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"[Stooq] Error descargando {ticker}: {e}")
        return pd.DataFrame()
    
    if df.empty:
        return pd.DataFrame()
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    df = df.loc[start:end]
    if df.empty:
        return pd.DataFrame()
    
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    return df[['open', 'high', 'low', 'close', 'volume']]

# =========================================================
# FUNCIÓN DE DESCARGA MASIVA CON REINTENTOS (YAHOO)
# =========================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def download_with_retry(tickers, start, end):
    return yf.download(
        tickers,
        start=start,
        end=end,
        interval='1d',
        group_by='ticker',
        auto_adjust=True,
        progress=False,
        threads=False
    )

# =========================================================
# CONSTRUCCIÓN DEL DATAFRAME FINAL (EVITA DUPLICACIÓN)
# =========================================================

def build_market_dataframe(prices, volumes, opens, highs, lows, all_tickers):
    """
    Construye el DataFrame final con columnas:
    - precio close
    - _volume, _open, _high, _low
    - _dollar_vol, _dollar_vol_smoothed
    """
    # Construir DataFrame eficientemente con concat
    df_list = [prices]
    for ticker in all_tickers:
        if ticker in prices.columns:
            df_list.append(volumes[[ticker]].rename(columns={ticker: f"{ticker}_volume"}))
            df_list.append(opens[[ticker]].rename(columns={ticker: f"{ticker}_open"}))
            df_list.append(highs[[ticker]].rename(columns={ticker: f"{ticker}_high"}))
            df_list.append(lows[[ticker]].rename(columns={ticker: f"{ticker}_low"}))
            # Calcular dollar_vol y smoothed
            dollar_vol = prices[ticker] * volumes[ticker]
            df_list.append(dollar_vol.to_frame(name=f"{ticker}_dollar_vol"))
            smoothed = dollar_vol.rolling(3, min_periods=1).mean()
            df_list.append(smoothed.to_frame(name=f"{ticker}_dollar_vol_smoothed"))
    df = pd.concat(df_list, axis=1)
    df = df.dropna(how='all')
    return df.dropna(how='all')

# =========================================================
# FUNCIÓN PRINCIPAL
# =========================================================

def download_market_data(force=False):
    """
    Descarga datos de mercado con:
    - caché de 23 horas
    - prioridad Yahoo (descarga masiva con reintentos)
    - fallback a Stooq (descarga por ticker)
    """
    cache_file = 'data/market_data.csv'
    if not force and os.path.exists(cache_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - file_time < timedelta(hours=23):
            print(f"Usando datos cacheados ({cache_file}) - menos de 23 horas.")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    all_tickers = ([tickers['benchmark'], tickers['bonds'], tickers['vix']] +
                   tickers['sectors'] +
                   [tickers['global'], "QQQ"] +
                   tickers['credit'] + tickers['rates'])
    end = datetime.now()
    start = end - timedelta(days=3650)

    # --- Intento con Yahoo (descarga masiva) ---
    try:
        data = download_with_retry(all_tickers, start, end)
        if data is not None and not data.empty:
            print("Datos descargados desde Yahoo Finance.")
            prices = pd.DataFrame()
            volumes = pd.DataFrame()
            opens = pd.DataFrame()
            highs = pd.DataFrame()
            lows = pd.DataFrame()
            for ticker in all_tickers:
                if ticker in data.columns.levels[0]:
                    prices[ticker] = data[ticker]['Close']
                    volumes[ticker] = data[ticker]['Volume']
                    opens[ticker] = data[ticker]['Open']
                    highs[ticker] = data[ticker]['High']
                    lows[ticker] = data[ticker]['Low']
            df = build_market_dataframe(prices, volumes, opens, highs, lows, all_tickers)
            df.to_csv(cache_file)
            print(f"Datos guardados desde Yahoo ({len(df)} días)")
            return df
    except Exception as e:
        print(f"Yahoo Finance falló: {e}. Intentando con Stooq como fallback...")

    # --- Fallback a Stooq (descarga por ticker) ---
    print("Descargando desde Stooq (puede tardar varios minutos)...")
    prices = pd.DataFrame()
    volumes = pd.DataFrame()
    opens = pd.DataFrame()
    highs = pd.DataFrame()
    lows = pd.DataFrame()
    for ticker in all_tickers:
        stooq_df = fetch_stooq(ticker, start, end)
        if stooq_df.empty:
            print(f"[Stooq] Advertencia: {ticker} no disponible.")
            continue
        prices[ticker] = stooq_df['close']
        volumes[ticker] = stooq_df['volume']
        opens[ticker] = stooq_df['open']
        highs[ticker] = stooq_df['high']
        lows[ticker] = stooq_df['low']
        print(f"[Stooq] {ticker} descargado correctamente.")
    if prices.empty:
        raise Exception("No se pudo descargar ningún ticker desde Stooq.")
    
    df = build_market_dataframe(prices, volumes, opens, highs, lows, all_tickers)
    df.to_csv(cache_file)
    print(f"Datos guardados desde Stooq ({len(df)} días)")
    return df

if __name__ == '__main__':
    download_market_data()