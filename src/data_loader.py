import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys, os
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import tickers

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

    try:
        data = download_with_retry(all_tickers, start, end)
    except Exception as e:
        print(f"Error tras reintentos: {e}")
        if os.path.exists(cache_file):
            print("Usando caché anterior como fallback.")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        raise

    # DataFrames para cada OHLCV
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
    print(f"Datos guardados con OHLCV y dollar volume ({len(df)} días)")
    return df

if __name__ == '__main__':
    download_market_data()