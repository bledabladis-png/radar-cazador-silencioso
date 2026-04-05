import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import tickers

def download_market_data():
    all_tickers = [tickers['benchmark'], tickers['bonds'], tickers['vix']] + tickers['sectors'] + [tickers['global']] + ["QQQ"] + tickers['credit'] + tickers['rates']
    end = datetime.now()
    start = end - timedelta(days=3650)
    
    data = yf.download(all_tickers, start=start, end=end, interval='1d', group_by='ticker', auto_adjust=True, progress=False)
    
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
    df.to_csv('data/market_data.csv')
    print(f"Datos guardados con volumen y dollar volume ({len(df)} días)")
    return df

if __name__ == '__main__':
    download_market_data()

