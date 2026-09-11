import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

HORIZONS = [5, 10, 20]
N_BOOTSTRAP = 1000
CFTC_RAW_CSV = Path('data/cache/cftc_tff.csv')

CONTRACT_TICKER = {
    'S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE': '^GSPC',
    'NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE': '^NDX',
    'RUSSELL 2000 MINI INDEX FUTURE - ICE FUTURES U.S.': '^RUT',
    'DJIA Consolidated - CHICAGO BOARD OF TRADE': '^DJI',
}

TARGET_CONTRACTS = list(CONTRACT_TICKER.keys())
PARTICIPANT_COLS = {
    'asset_mgr': ('Asset_Mgr_Positions_Long_All', 'Asset_Mgr_Positions_Short_All',
                  'Change_in_Asset_Mgr_Long_All', 'Change_in_Asset_Mgr_Short_All'),
    'lev_money': ('Lev_Money_Positions_Long_All', 'Lev_Money_Positions_Short_All',
                  'Change_in_Lev_Money_Long_All', 'Change_in_Lev_Money_Short_All'),
    'dealer': ('Dealer_Positions_Long_All', 'Dealer_Positions_Short_All',
               'Change_in_Dealer_Long_All', 'Change_in_Dealer_Short_All'),
}

print('Leyendo CFTC crudo desde caché...')
df = pd.read_csv(CFTC_RAW_CSV, low_memory=False)
df = df[df['Market_and_Exchange_Names'].isin(TARGET_CONTRACTS)].copy()
df['date'] = pd.to_datetime(df['Report_Date_as_YYYY_MM_DD'], errors='coerce', format='mixed')
df = df.dropna(subset=['date'])

# Limpiar y convertir columnas numéricas
numeric_cols = []
for cols in PARTICIPANT_COLS.values():
    numeric_cols.extend(cols)
for col in numeric_cols:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace(',', '', regex=False)
            .str.replace(' ', '', regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

print(f'Filas CFTC seleccionadas: {len(df)}')

# Construir DataFrame largo con todas las observaciones
long_frames = []
for participant, (long_col, short_col, chg_long, chg_short) in PARTICIPANT_COLS.items():
    if long_col not in df.columns or short_col not in df.columns:
        continue
    temp = pd.DataFrame({
        'date': df['date'].values,
        'contract': df['Market_and_Exchange_Names'].values,
        'participant': participant,
        'net_position': df[long_col] - df[short_col],
    })
    if chg_long in df.columns and chg_short in df.columns:
        pos_change = df[chg_long] - df[chg_short]
        # Rellenar NaN con diff
        for contract, group in temp.groupby('contract'):
            idx = group.index
            diff = temp.loc[idx, 'net_position'].diff()
            pos_change.loc[idx] = pos_change.loc[idx].fillna(diff)
        temp['position_change'] = pos_change.values
    else:
        temp['position_change'] = temp.groupby('contract')['net_position'].diff()

    # Calcular flow_z
    temp = temp.sort_values(['contract', 'date'])
    rolling_mean = temp.groupby('contract')['position_change'].transform(lambda x: x.rolling(52, min_periods=10).mean())
    rolling_std = temp.groupby('contract')['position_change'].transform(lambda x: x.rolling(52, min_periods=10).std())
    temp['flow_z'] = ((temp['position_change'] - rolling_mean) / (rolling_std + 1e-9)).fillna(0.0)
    long_frames.append(temp)

flow_all = pd.concat(long_frames, ignore_index=True)
flow_all = flow_all.dropna(subset=['position_change', 'net_position'])
flow_all = flow_all.sort_values('date').reset_index(drop=True)

print(f'Observaciones totales de flujo CFTC: {len(flow_all)}')

print('Descargando precios de índices...')
session = None
try:
    from curl_cffi import requests as curl_requests
    session = curl_requests.Session(impersonate='chrome')
except:
    pass

prices = {}
for ticker in CONTRACT_TICKER.values():
    try:
        if session:
            data = yf.download(ticker, period='2y', auto_adjust=True, progress=False, session=session)
        else:
            data = yf.download(ticker, period='2y', auto_adjust=True, progress=False)
        close = data['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if close.empty:
            continue
        prices[ticker] = close
    except Exception as e:
        print(f'  Error descargando {ticker}: {e}')

print(f'Precios descargados para {len(prices)} índices.')

results = []
for participant in flow_all['participant'].unique():
    sub = flow_all[flow_all['participant'] == participant]
    for contract, ticker in CONTRACT_TICKER.items():
        if ticker not in prices:
            continue
        price = prices[ticker]
        flow_t = sub[sub['contract'] == contract].set_index('date')['flow_z']
        flow_t = flow_t[~flow_t.index.duplicated(keep='first')]

        for h in HORIZONS:
            target = price.pct_change(h).shift(-h)
            common_idx = flow_t.index.intersection(target.index)
            if len(common_idx) < 30:
                continue
            x = flow_t.loc[common_idx]
            y = target.loc[common_idx]
            if isinstance(x, pd.DataFrame): x = x.iloc[:, 0]
            if isinstance(y, pd.DataFrame): y = y.iloc[:, 0]
            mask = (~(x.isna() | y.isna())).astype(bool)
            x = x[mask]
            y = y[mask]
            if len(x) < 30:
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
                'contract': contract,
                'participant': participant,
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
    out = Path('outputs/audit/oos_cftc_results.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out, index=False)
    print(f'\nResultados guardados en {out}')
    print(df_res[['contract','participant','horizon','rho_spearman','p_bonferroni','significant','n_obs']].to_string(index=False))
else:
    print('No se generaron resultados. Revisar datos.')
