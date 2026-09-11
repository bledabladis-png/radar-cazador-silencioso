import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

HORIZONS = [5, 10, 20]
N_BOOTSTRAP = 1000
TICKERS = ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLRE','XLU','XLC','FEZ']
FLOW_CSV = Path('outputs/history/etf_primary_flow.csv')

print('Cargando flujos primarios...')
flows = pd.read_csv(FLOW_CSV, parse_dates=['Date'])
flows = flows.rename(columns={'Date':'date'})
flows = flows[flows['ticker'].isin(TICKERS)]
flows = flows.dropna(subset=['primary_flow_z'])
flows['date'] = pd.to_datetime(flows['date'])

print(f'Total filas históricas de flujo: {len(flows)}')

print('Descargando precios OHLCV con Yahoo Finance...')
session = None
try:
    from curl_cffi import requests as curl_requests
    session = curl_requests.Session(impersonate='chrome')
except:
    pass

prices = {}
for ticker in TICKERS:
    try:
        if session:
            data = yf.download(ticker, period='2y', auto_adjust=True, progress=False, session=session)
        else:
            data = yf.download(ticker, period='2y', auto_adjust=True, progress=False)
        # Extraer cierre como Serie
        close = data['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]  # tomar la única columna
        close = close.dropna()
        if close.empty:
            continue
        prices[ticker] = close
    except Exception as e:
        print(f'  Error descargando {ticker}: {e}')

print(f'Precios descargados para {len(prices)} tickers.')

results = []
for ticker in TICKERS:
    if ticker not in prices:
        continue
    price = prices[ticker]
    # Forward returns como Series
    fwd = {}
    for h in HORIZONS:
        fwd[h] = price.pct_change(h).shift(-h)  # Series

    flow_t = flows[flows['ticker'] == ticker].set_index('date')['primary_flow_z']
    flow_t = flow_t[~flow_t.index.duplicated(keep='first')]

    for h in HORIZONS:
        target = fwd[h]
        common_idx = flow_t.index.intersection(target.index)
        if len(common_idx) < 30:
            continue
        x = flow_t.loc[common_idx]
        y = target.loc[common_idx]

        # Asegurar que x y y son Series
        if isinstance(x, pd.DataFrame):
            x = x.iloc[:, 0]
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        # Construir mask booleano como Series
        mask = (~(x.isna() | y.isna())).astype(bool)

        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 30:
            continue

        rho, pval = spearmanr(x_clean, y_clean)

        boot_rhos = []
        for _ in range(N_BOOTSTRAP):
            idx = np.random.choice(len(x_clean), len(x_clean), replace=True)
            r_b, _ = spearmanr(x_clean.iloc[idx], y_clean.iloc[idx])
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
            'n_obs': len(x_clean),
        })

df_res = pd.DataFrame(results)
df_res['p_bonferroni'] = df_res['p_value'] * len(HORIZONS)
df_res['significant'] = df_res['p_bonferroni'] < 0.05

out = Path('outputs/audit/oos_flow_results.csv')
out.parent.mkdir(parents=True, exist_ok=True)
df_res.to_csv(out, index=False)
print(f'\nResultados guardados en {out}')
print(df_res[['ticker','horizon','rho_spearman','p_bonferroni','significant','n_obs']].to_string(index=False))
