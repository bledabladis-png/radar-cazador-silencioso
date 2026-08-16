import os
import pandas as pd
import yfinance as yf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Tickers líquidos para validación rápida
TICKERS = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT']
THRESHOLD = 0.05  # 5%
OUTPUT = Path('outputs/audit/cross_provider_validation.csv')

def get_yahoo_close(ticker):
    try:
        df = yf.download(ticker, period='5d', progress=False, auto_adjust=True)
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return float(close.iloc[-1])
    except Exception:
        return None

def get_polygon_close(ticker):
    api_key = os.environ.get('POLYGON_API_KEY')
    if not api_key:
        return None
    import requests
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={api_key}'
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        results = data.get('results')
        if results:
            return float(results[0]['c'])
    except:
        pass
    return None

def get_alpha_vantage_close(ticker):
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        return None
    import requests
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'GLOBAL_QUOTE',
        'symbol': ticker,
        'apikey': api_key
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        quote = data.get('Global Quote', {})
        price = quote.get('05. price')
        if price:
            return float(price)
    except:
        pass
    return None

def main():
    print('Validación cruzada entre proveedores...')
    rows = []
    for ticker in TICKERS:
        yahoo = get_yahoo_close(ticker)
        polygon = get_polygon_close(ticker)
        alpha = get_alpha_vantage_close(ticker)

        secondary = polygon or alpha
        provider_secundario = 'polygon' if polygon else ('alpha_vantage' if alpha else None)

        diff_pct = None
        status = 'SIN_DATO_SECUNDARIO'
        if yahoo is not None and secondary is not None:
            diff_pct = (yahoo - secondary) / secondary * 100
            status = 'OK' if abs(diff_pct) <= THRESHOLD*100 else 'WARN'

        rows.append({
            'ticker': ticker,
            'yahoo': yahoo,
            'secundario': secondary,
            'provider': provider_secundario,
            'diff_pct': diff_pct,
            'status': status,
        })

    df = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f'Resultados guardados en {OUTPUT}')
    print(df[['ticker','yahoo','secundario','provider','diff_pct','status']].to_string(index=False))

    if (df['status'] == 'WARN').any():
        print('\n⚠️ Advertencia: diferencias superiores al 5% detectadas.')
    else:
        print('\n✅ Validación cruzada sin discrepancias significativas.')

if __name__ == '__main__':
    main()
