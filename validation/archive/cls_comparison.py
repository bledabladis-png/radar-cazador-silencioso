# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.utils import get_col
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("COMPARACIÓN CLS v1.0 vs CLS v1.1")
print("=" * 70)

# Cargar datos de mercado
df_market = pd.read_csv('data/market_data.csv', header=[0,1], index_col=0, parse_dates=True)

# Cargar NFCI y Credit OAS desde FRED
nfci = pd.read_csv('data/macro_manual/nfci.csv', index_col=0, parse_dates=True)['NFCI']
oas = pd.read_csv('data/macro_manual/credit_oas.csv', index_col=0, parse_dates=True)['CreditOAS']

# Fechas de evaluación (520 semanas)
start_date = df_market.index[0]
end_date = df_market.index[-1]
eval_dates = pd.date_range(start_date, end_date, freq='W-FRI')
eval_dates = [d for d in eval_dates if d in df_market.index]

print(f"Evaluando {len(eval_dates)} semanas...")

def robust_zscore_series(series, window=104):
    median = series.rolling(window, min_periods=20).median()
    def mad_func(x):
        return np.median(np.abs(x - np.median(x)))
    mad = series.rolling(window, min_periods=20).apply(mad_func, raw=True)
    return (series - median) / (1.4826 * mad + 1e-9)

def stress_transform(z):
    return float(np.clip(np.tanh(z.iloc[-1] / 2.0), 0, 1)) if len(z) > 0 else 0.5

cls_v10 = []
cls_v11 = []

for i, date in enumerate(eval_dates):
    if i % 50 == 0:
        print(f"  Progreso: {i}/{len(eval_dates)}")
    idx = df_market.index.get_loc(date)
    df_slice = df_market.iloc[:idx+1]
    
    try:
        # --- CLS v1.0 (sintético) ---
        vix_close = get_col(df_slice, '^VIX', 'Close')
        vix_ret = vix_close.pct_change().rolling(20).std().iloc[-1]
        vix_ma = vix_close.pct_change().rolling(60).std().mean()
        fc_approx = float(np.clip(np.tanh((vix_ret / (vix_ma + 1e-9) - 1) / 2), 0, 1))
        
        hyg = get_col(df_slice, 'HYG', 'Close')
        lqd = get_col(df_slice, 'LQD', 'Close')
        spread = hyg / lqd
        cred_approx = float(np.clip(np.tanh(-(spread.pct_change(20).iloc[-1]) / 2), 0, 1))
        
        cls_v10_val = float(np.sqrt(np.mean(np.square([fc_approx, cred_approx, 0.3, 0.3]))))
        cls_v10.append(cls_v10_val)
        
        # --- CLS v1.1 (con FRED) ---
        if date in nfci.index:
            nfci_window = nfci.loc[:date]
            nfci_z = robust_zscore_series(nfci_window)
            nfci_stress = stress_transform(nfci_z)
        else:
            nfci_stress = 0.5
        
        if date in oas.index:
            oas_window = oas.loc[:date]
            oas_z = robust_zscore_series(oas_window)
            oas_stress = stress_transform(oas_z)
        else:
            oas_stress = 0.5
        
        credit_family = 0.60 * oas_stress + 0.40 * fc_approx
        vix_stress = float(np.clip(np.tanh(vix_ret / 2), 0, 1))
        
        cls_v11_val = (0.25 * nfci_stress +
                       0.35 * credit_family +
                       0.25 * vix_stress +
                       0.15 * 0.5)  # complementarios a 0.5
        cls_v11.append(cls_v11_val)
        
    except Exception:
        cls_v10.append(np.nan)
        cls_v11.append(np.nan)

df_cls = pd.DataFrame({
    'date': eval_dates,
    'CLS_v10': cls_v10,
    'CLS_v11': cls_v11
}).dropna()

print(f"\nRegistros válidos: {len(df_cls)}")

# Estadísticas comparativas
print("\n" + "="*70)
print("ESTADÍSTICAS COMPARATIVAS")
print("="*70)
for col in ['CLS_v10', 'CLS_v11']:
    s = df_cls[col]
    print(f"\n{col}:")
    print(f"  Media: {s.mean():.4f}  Mediana: {s.median():.4f}  Max: {s.max():.4f}")
    print(f"  P90: {s.quantile(0.90):.4f}  P95: {s.quantile(0.95):.4f}  P99: {s.quantile(0.99):.4f}")

# Correlación
corr = df_cls['CLS_v10'].corr(df_cls['CLS_v11'])
print(f"\nCorrelación Spearman: {corr:.4f}")

# Diferencia media
delta = df_cls['CLS_v11'] - df_cls['CLS_v10']
print(f"\nDelta (v11 - v10): media={delta.mean():.4f}, std={delta.std():.4f}")

# Marzo 2020
for date_str in ['2020-03-16', '2020-03-23']:
    if date_str in df_cls['date'].values:
        row = df_cls[df_cls['date'] == date_str].iloc[0]
        print(f"\n{date_str}:")
        print(f"  CLS v1.0: {row['CLS_v10']:.4f}")
        print(f"  CLS v1.1: {row['CLS_v11']:.4f}")

print("\n" + "="*70)
print("COMPARACIÓN COMPLETADA")
print("="*70)
