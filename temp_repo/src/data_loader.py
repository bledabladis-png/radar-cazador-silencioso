import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
import os
from config.tickers import MARKET_TICKERS
from config.settings import CACHE_HOURS

def _ticker_list():
    tickers = []
    for group in MARKET_TICKERS.values():
        if isinstance(group, dict):
            tickers.extend(group.values())
        elif isinstance(group, list):
            tickers.extend(group)
    return list(set(tickers))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_market_data():
    cache_path = 'data/market_data.csv'
    if os.path.exists(cache_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - mtime < timedelta(hours=CACHE_HOURS):
            return pd.read_csv(cache_path, header=[0,1], index_col=0, parse_dates=True)

    tickers = _ticker_list()
    data = yf.download(tickers, period='10y', auto_adjust=True)

    if not isinstance(data.columns, pd.MultiIndex):
        data.columns = pd.MultiIndex.from_tuples(data.columns)

    from src.utils import clean_oil_prices
    data = clean_oil_prices(data)

    data.to_csv(cache_path)
    return data
