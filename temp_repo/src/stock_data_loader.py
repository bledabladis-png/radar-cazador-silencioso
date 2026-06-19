import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
import os

CACHE_HOURS = 23

def get_stock_list():
    try:
        df = pd.read_csv('data/etf_holdings.csv')
        return df['ticker'].unique().tolist()
    except Exception:
        return []

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

    data = yf.download(tickers, period='5y', auto_adjust=True)
    if not isinstance(data.columns, pd.MultiIndex):
        data.columns = pd.MultiIndex.from_tuples(data.columns)

    data.to_csv(cache_path)
    return data
