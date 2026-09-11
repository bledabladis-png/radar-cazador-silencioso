import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

HORIZONS = [20, 40, 60]
N_BOOTSTRAP = 1000
NportCSV = Path('outputs/history/sec_nport_position_change_quarterly.csv')
HOLDINGS_SECTOR = Path('data/etf_holdings.csv')
HOLDINGS_INDEX = Path('data/index_holdings.csv')

print('Cargando cambios N-PORT...')
nport = pd.read_csv(NportCSV, parse_dates=['REPORT_DATE'])
nport = nport.dropna(subset=['POSITION_CHANGE'])
nport['REPORT_DATE'] = pd.to_datetime(nport['REPORT_DATE'])

print('Cargando holdings sectoriales e índices...')
sec = pd.read_csv(HOLDINGS_SECTOR)[['ticker','identifier']]
idx = pd.read_csv(HOLDINGS_INDEX)[['ticker','identifier']]
holdings = pd.concat([sec, idx], ignore_index=True)
holdings = holdings.dropna(subset=['identifier'])
holdings['identifier'] = holdings['identifier'].str.upper().str.strip()

# Crear mapa CUSIP -> ticker
cusip_ticker = dict(zip(holdings['identifier'], holdings['ticker']))

# Filtrar N-PORT por CUSIP presente en holdings
nport['cusip_clean'] = nport['ISSUER_CUSIP'].str.upper().str.strip()
nport_filtered = nport[nport['cusip_clean'].isin(cusip_ticker.keys())].copy()
nport_filtered['ticker'] = nport_filtered['cusip_clean'].map(cusip_ticker)

print(f'Registros N-PORT con CUSIP en holdings: {len(nport_filtered)}')
print(f'Tickers únicos con datos: {nport_filtered["ticker"].nunique()}')

tickers = nport_filtered['ticker'].unique()
print('Descargando precios OHLCV...')
session = None
try:
    from curl_cffi import requests as curl_requests
    session = curl_requests.Session(impersonate='chrome')
except:
    pass

prices = {}
for ticker in tickers:
    try:
        if session:
            data = yf.download(ticker, period='2y', auto_adjust=True, progress=False, session=session)
        else:
            data = yf.download(ticker, period='2y', auto_adjust=True, progress=False)
        close = data['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if not close.empty:
            prices[ticker] = close
    except Exception as e:
        print(f'  Error descargando {ticker}: {e}')

print(f'Precios descargados para {len(prices)} tickers')

results = []
for ticker, price in prices.items():
    sub = nport_filtered[nport_filtered['ticker'] == ticker].set_index('REPORT_DATE')
    if 'POSITION_CHANGE_PCT' in sub.columns:
        metric = sub['POSITION_CHANGE_PCT']
    else:
        metric = sub['POSITION_CHANGE']
    metric = metric[~metric.index.duplicated(keep='first')]
    metric = metric.sort_index()

    for h in HORIZONS:
        target = price.pct_change(h).shift(-h)
        common_idx = metric.index.intersection(target.index)
        if len(common_idx) < 10:
            continue
        x = metric.loc[common_idx]
        y = target.loc[common_idx]
        mask = (~(x.isna() | y.isna())).astype(bool)
        x = x[mask]
        y = y[mask]
        if len(x) < 10:
            continue
        rho, pval = spearmanr(x, y)

        boot_rhos = []
        for _ in range(N_BOOTSTRAP):
            idx = np.random.choice(len(x), len(x), replace=True)
            r_b, _ = spearmanr(x.iloc[idx], y.iloc[idx])
            boot_rhos.append(r_b)
        ci_low = np.percentile(boot_rhos, 2.5)
        ci_high = np.percentile(boot_rhos, 97.5)
        results.append({
            'ticker': ticker,
            'horizon': h,
            'rho_spearman': rho,
            'p_value': pval,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n_obs': len(x),
        })

if results:
    df_res = pd.DataFrame(results)
    df_res['p_bonferroni'] = df_res['p_value'] * len(HORIZONS)
    df_res['significant'] = df_res['p_bonferroni'] < 0.05
    out = Path('outputs/audit/oos_nport_results.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out, index=False)
    print(f'\nResultados guardados en {out}')
    print(df_res[['ticker','horizon','rho_spearman','p_bonferroni','significant','n_obs']].to_string(index=False))
else:
    print('No se generaron resultados. Insuficientes datos.')
