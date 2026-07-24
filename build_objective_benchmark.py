import pandas as pd
import numpy as np
import yfinance as yf
from data.providers.fred import FredProvider

print("Descargando SPY, HYG, LQD, ^VIX...")
prices = yf.download(['SPY', 'HYG', 'LQD', '^VIX'], start='2006-01-01', auto_adjust=True)['Close']

# Resample a fin de mes
monthly = prices.resample('ME').last()
# Calcular retornos
monthly['spy_ret_1y'] = monthly['SPY'].pct_change(12, fill_method=None)
monthly['spy_ret_6m'] = monthly['SPY'].pct_change(6, fill_method=None)
# Ratio HYG/LQD y su cambio mensual
monthly['hyg_lqd'] = monthly['HYG'] / monthly['LQD']
monthly['hyg_lqd_chg'] = monthly['hyg_lqd'].pct_change(fill_method=None)

# VIX (media mensual)
vix = yf.download('^VIX', start='2006-01-01')['Close']
monthly['vix'] = vix.resample('ME').mean()

# CPI y desempleo desde FRED
fred = FredProvider()
cpi = fred._download_series('CPIAUCSL', start='2006-01-01')
cpi_yoy = cpi.pct_change(12, fill_method=None) * 100
cpi_yoy.name = 'cpi_yoy'
unrate = fred._download_series('UNRATE', start='2006-01-01')
unrate.name = 'unemployment'

# Unir al DataFrame mensual
monthly = monthly.join(cpi_yoy.resample('ME').last())
monthly = monthly.join(unrate.resample('ME').last())

# Fechas de recesión NBER
nber_recessions = [
    ('2007-12-01', '2009-06-30'),
    ('2020-02-01', '2020-04-30')
]

print("Aplicando reglas...")
results = []
for d in monthly.index:
    if d < pd.Timestamp('2007-01-01'):  # sin suficientes datos anteriores
        results.append({'date': d.strftime('%Y-%m-%d'), 'regime': 'MIXED'})
        continue

    row = monthly.loc[d]
    spy_1y = row['spy_ret_1y'] if pd.notna(row['spy_ret_1y']) else 0
    spy_6m = row['spy_ret_6m'] if pd.notna(row['spy_ret_6m']) else 0
    hyg_chg = row['hyg_lqd_chg'] if pd.notna(row['hyg_lqd_chg']) else 0
    vix_val = row['vix'] if pd.notna(row['vix']) else 0
    cpi_val = row['cpi_yoy'] if pd.notna(row['cpi_yoy']) else 0
    unemp = row['unemployment'] if pd.notna(row['unemployment']) else 0

    regime = 'MIXED'

    # 1. Liquidity Crisis
    if vix_val > 40 or (vix_val > 30 and hyg_chg < -0.05):
        regime = 'LIQUIDITY CRISIS'
    # 2. Recession (NBER)
    elif any(pd.Timestamp(s) <= d <= pd.Timestamp(e) for s, e in nber_recessions):
        regime = 'RECESSION'
    # 3. Inflation Shock
    elif cpi_val > 5 and spy_6m < -0.10:
        regime = 'INFLATION SHOCK'
    # 4. Expansion
    elif spy_1y > 0.15 and vix_val < 25:
        regime = 'EXPANSION'
    # 5. Recovery
    elif spy_1y > 0.08 and unemp < 5:
        regime = 'RECOVERY'
    # 6. Stagflation
    elif cpi_val > 4 and -0.05 <= spy_1y <= 0.05:
        regime = 'STAGFLATION'
    # 7. Slowdown
    elif -0.15 < spy_1y < -0.05:
        regime = 'SLOWDOWN'

    results.append({'date': d.strftime('%Y-%m-%d'), 'regime': regime})

# Convertir a intervalos
df = pd.DataFrame(results)
df['date'] = pd.to_datetime(df['date'])
df['regime_shift'] = (df['regime'] != df['regime'].shift()).cumsum()
intervals = []
for _, group in df.groupby('regime_shift'):
    start = group['date'].iloc[0].strftime('%Y-%m-%d')
    end = group['date'].iloc[-1].strftime('%Y-%m-%d')
    regime = group['regime'].iloc[0]
    intervals.append((start, end, regime))

out_df = pd.DataFrame(intervals, columns=['start', 'end', 'regime'])
out_df.to_csv('data/expected_regimes_objective.csv', index=False)
print("\nBenchmark objetivo guardado en data/expected_regimes_objective.csv")
print(f"Total de intervalos: {len(out_df)}")
print(out_df.to_string())
