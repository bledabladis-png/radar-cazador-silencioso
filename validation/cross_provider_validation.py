import os
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data.providers.yahoo import YahooProvider

TICKERS = ['SPY', 'AAPL']
OUTPUT = Path('outputs/audit/cross_provider_validation.csv')

def get_yahoo_close(ticker):
    try:
        provider = YahooProvider()
        df = provider.get_prices([ticker], period='5d')
        if df is None or df.empty:
            return None
        close_cols = [c for c in df.columns if c[0] == 'Close']
        if not close_cols:
            return None
        return float(df[close_cols[0]].iloc[-1])
    except Exception as e:
        print(f'  Yahoo error para {ticker}: {e}')
        return None

def main():
    print('Validación de YahooProvider en CI...')
    rows = []
    for ticker in TICKERS:
        yahoo = get_yahoo_close(ticker)
        status = 'OK' if yahoo is not None else 'ERROR'
        rows.append({'ticker': ticker, 'yahoo': yahoo, 'status': status})
    df = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(df.to_string(index=False))
    if (df['status'] == 'ERROR').any():
        print('\n⚠️ Se detectaron errores al obtener datos de YahooProvider.')
    else:
        print('\n✅ YahooProvider funciona correctamente.')

if __name__ == '__main__':
    main()
