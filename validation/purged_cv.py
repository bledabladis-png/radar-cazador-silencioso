import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import accuracy_score

# Cargar resultados del backtest
df = pd.read_csv('outputs/audit/backtest_v3_results.csv', parse_dates=['date'])
df = df.set_index('date').sort_index()

# Retornos del SPY
spy = yf.download('^GSPC', period='20y', auto_adjust=True)['Close']
spy_ret = spy.resample('W-FRI').last().pct_change().dropna()
spy_ret.name = 'spy_return'

common = df.index.intersection(spy_ret.index)
df = df.loc[common].copy()
df['spy_return'] = spy_ret.loc[common]

# Configuración de Purged K-Fold
K = 5
n = len(df)
purge_pct = 0.05  # 5% de purga a cada lado del límite
indices = np.arange(n)
fold_size = n // K
results = []

for k in range(K):
    # Definir límites del fold de prueba
    test_start = k * fold_size
    test_end = (k + 1) * fold_size if k < K - 1 else n
    test_idx = indices[test_start:test_end]
    
    # Purga: eliminar observaciones cercanas a los límites
    purge_margin = int(n * purge_pct)
    train_idx = []
    for i in range(K):
        if i == k:
            continue
        fold_start = i * fold_size
        fold_end = (i + 1) * fold_size if i < K - 1 else n
        train_idx.extend(indices[fold_start + purge_margin : fold_end - purge_margin])
    
    train_idx = np.array(train_idx)
    
    # Métricas en el fold de prueba
    test_df = df.iloc[test_idx]
    if len(test_df) < 20:
        continue
    
    acc = accuracy_score(test_df['expected'], test_df['obtained'])
    
    # Sharpe de estrés y positivo en este fold
    stress_regimes = ['LIQUIDITY CRISIS', 'RECESSION', 'SLOWDOWN']
    pos_regimes = ['EXPANSION', 'RECOVERY']
    ret_stress = test_df[test_df['obtained'].isin(stress_regimes)]['spy_return']
    ret_pos = test_df[test_df['obtained'].isin(pos_regimes)]['spy_return']
    
    sharpe_stress = (ret_stress.mean() / (ret_stress.std() + 1e-9)) * np.sqrt(52) if len(ret_stress) > 1 else np.nan
    sharpe_pos = (ret_pos.mean() / (ret_pos.std() + 1e-9)) * np.sqrt(52) if len(ret_pos) > 1 else np.nan
    
    results.append({
        'fold': k + 1,
        'accuracy': acc,
        'sharpe_stress': sharpe_stress,
        'sharpe_pos': sharpe_pos,
        'separacion': sharpe_pos - sharpe_stress if pd.notna(sharpe_pos) and pd.notna(sharpe_stress) else np.nan
    })

res_df = pd.DataFrame(results)
print("=== PURGED K‑FOLD CROSS‑VALIDATION ===")
print(f"K = {K}, purga = {purge_pct*100:.0f}%\n")
print(res_df.to_string(float_format=lambda x: f'{x:.2f}'))
print(f"\nPrecisión media: {res_df['accuracy'].mean():.2%} ± {res_df['accuracy'].std():.2%}")
print(f"Separación media: {res_df['separacion'].mean():.2f} ± {res_df['separacion'].std():.2f}")
