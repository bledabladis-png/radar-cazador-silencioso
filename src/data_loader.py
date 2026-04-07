import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys, os
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import tickers

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def download_with_retry(tickers, start, end):
    """Descarga con reintentos automáticos ante fallos temporales."""
    return yf.download(
        tickers,
        start=start,
        end=end,
        interval='1d',
        group_by='ticker',
        auto_adjust=True,
        progress=False,
        threads=False   # Evita saturación de conexiones
    )

def download_market_data(force=False):
    """
    Descarga datos de mercado con caché temporal y reintentos.
    Si force=True, ignora la caché y fuerza descarga.
    """
    cache_file = 'data/market_data.csv'
    
    # 1. Intentar usar caché si es reciente
    if not force and os.path.exists(cache_file):
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - file_time < timedelta(hours=23):
            print(f"Usando datos cacheados ({cache_file}) - menos de 23 horas.")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
    
    # 2. Si no hay caché válida, descargar
    all_tickers = ([tickers['benchmark'], tickers['bonds'], tickers['vix']] +
                   tickers['sectors'] +
                   [tickers['global'], "QQQ"] +
                   tickers['credit'] + tickers['rates'])
    end = datetime.now()
    start = end - timedelta(days=3650)
    
    try:
        data = download_with_retry(all_tickers, start, end)
    except Exception as e:
        print(f"Error tras reintentos: {e}")
        # Fallback: usar caché anterior aunque sea vieja
        if os.path.exists(cache_file):
            print("Usando caché anterior como fallback.")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        raise  # Si no hay caché, propagar el error
    
    # 3. Procesar datos
    prices = pd.DataFrame()
    volumes = pd.DataFrame()
    
    for ticker in all_tickers:
        if ticker in data.columns.levels[0]:
            prices[ticker] = data[ticker]['Close']
            volumes[ticker] = data[ticker]['Volume']
        else:
            if len(all_tickers) == 1:
                prices[ticker] = data['Close']
                volumes[ticker] = data['Volume']
    
    df = prices.copy()
    for ticker in all_tickers:
        df[f"{ticker}_volume"] = volumes[ticker]
        df[f"{ticker}_dollar_vol"] = prices[ticker] * volumes[ticker]
        df[f"{ticker}_dollar_vol_smoothed"] = (prices[ticker] * volumes[ticker]).rolling(3, min_periods=1).mean()
    
    df = df.dropna(how='all')
    df.to_csv(cache_file)
    print(f"Datos guardados con volumen y dollar volume ({len(df)} días)")
    return df

if __name__ == '__main__':
    download_market_data()