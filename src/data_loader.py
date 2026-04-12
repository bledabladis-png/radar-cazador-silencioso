import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys, os
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import tickers

def fetch_stooq(ticker, start, end):
    """
    Descarga datos OHLCV desde Stooq para un ticker.
    Retorna DataFrame con columnas: open, high, low, close, volume.
    """
    # Mapeo de tickers comunes a símbolos de Stooq
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
    
    # Renombrar al formato esperado
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })
    return df[['open', 'high', 'low', 'close', 'volume']]

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

def download_market_data(force=False):
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

    # Intentar primero con Yahoo Finance (descarga masiva)
    try:
        data = download_with_retry(all_tickers, start, end)
        if data is not None and not data.empty:
            print("Datos descargados desde Yahoo Finance.")
            # Procesar datos (extraer OHLCV)
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
                else:
                    if len(all_tickers) == 1:
                        prices[ticker] = data['Close']
                        volumes[ticker] = data['Volume']
                        opens[ticker] = data['Open']
                        highs[ticker] = data['High']
                        lows[ticker] = data['Low']
            # Construir DataFrame con columnas adicionales
            df = prices.copy()
            for ticker in all_tickers:
                df[f"{ticker}_volume"] = volumes[ticker]
                df[f"{ticker}_open"] = opens[ticker]
                df[f"{ticker}_high"] = highs[ticker]
                df[f"{ticker}_low"] = lows[ticker]
                df[f"{ticker}_dollar_vol"] = prices[ticker] * volumes[ticker]
                df[f"{ticker}_dollar_vol_smoothed"] = (prices[ticker] * volumes[ticker]).rolling(3, min_periods=1).mean()
            df = df.dropna(how='all')
            df.to_csv(cache_file)
            print(f"Datos guardados desde Yahoo ({len(df)} días)")
            return df
    except Exception as e:
        print(f"Yahoo Finance falló: {e}. Intentando con Stooq como fallback...")

    # Fallback a Stooq (descarga por ticker, más lento pero robusto)
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
    
    # Construir DataFrame igual que con Yahoo
    df = prices.copy()
    for ticker in all_tickers:
        if ticker in prices.columns:
            df[f"{ticker}_volume"] = volumes[ticker] if ticker in volumes.columns else np.nan
            df[f"{ticker}_open"] = opens[ticker] if ticker in opens.columns else np.nan
            df[f"{ticker}_high"] = highs[ticker] if ticker in highs.columns else np.nan
            df[f"{ticker}_low"] = lows[ticker] if ticker in lows.columns else np.nan
            df[f"{ticker}_dollar_vol"] = prices[ticker] * volumes[ticker]
            df[f"{ticker}_dollar_vol_smoothed"] = (prices[ticker] * volumes[ticker]).rolling(3, min_periods=1).mean()
    df = df.dropna(how='all')
    df.to_csv(cache_file)
    print(f"Datos guardados desde Stooq ({len(df)} días)")
    return df

if __name__ == '__main__':
    download_market_data()