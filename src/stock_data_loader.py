import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import time

CACHE_HOURS = 23

def get_stock_list():
    try:
        df = pd.read_csv('data/etf_holdings.csv')
        # Limitar a 20 tickers por sector, ordenando por weight descendente
        if 'weight' in df.columns:
            df = df.sort_values(['etf', 'weight'], ascending=[True, False])
        result = []
        for etf, group in df.groupby('etf'):
            result.extend(group['ticker'].head(20).tolist())
        return result
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

    batch_size = 5
    delay = 2
    all_data = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"Descargando lote {i//batch_size + 1}: {batch}")
        try:
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
