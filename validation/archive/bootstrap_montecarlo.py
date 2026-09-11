import pandas as pd
import numpy as np
import yfinance as yf

# Cargar resultados del backtest
df = pd.read_csv('outputs/audit/backtest_v2_results.csv', parse_dates=['date'])
df = df.set_index('date')

# Descargar SPY directamente
data = yf.download('^GSPC', period='20y', auto_adjust=True)
spy = data['Close'].resample('W-FRI').last().pct_change().dropna()
spy.name = 'spy_return'

# Alinear fechas (ambos son viernes)
common_dates = df.index.intersection(spy.index)
df_aligned = df.loc[common_dates].copy()
df_aligned['spy_return'] = spy.loc[common_dates]

regimes = df_aligned['obtained'].unique()
n_boot = 2000

print("=== BOOTSTRAP DEL SHARPE POR RÉGIMEN ===")
for regime in sorted(regimes):
    mask = df_aligned['obtained'] == regime
    returns = df_aligned.loc[mask, 'spy_return'].dropna()
    if len(returns) < 5:
        print(f"{regime}: insuficientes datos ({len(returns)} semanas)")
        continue
    sharpe_obs = returns.mean() / (returns.std() + 1e-9) * np.sqrt(52)
    boot_sharpes = []
    for _ in range(n_boot):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        boot_sharpes.append(sample.mean() / (sample.std() + 1e-9) * np.sqrt(52))
    ci_low, ci_high = np.percentile(boot_sharpes, [2.5, 97.5])
    print(f"{regime}: Sharpe obs={sharpe_obs:.2f}, IC 95% = [{ci_low:.2f}, {ci_high:.2f}]")

print("\n=== MONTECARLO DE EQUITY BAJO CADA RÉGIMEN ===")
n_sim = 500
for regime in sorted(regimes):
    mask = df_aligned['obtained'] == regime
    returns = df_aligned.loc[mask, 'spy_return'].dropna()
    if len(returns) < 5:
        continue
    sim_equity = np.zeros((n_sim, 52))
    for i in range(n_sim):
        idx = np.random.choice(len(returns), size=52, replace=True)
        sim_ret = returns.iloc[idx].values
        sim_equity[i] = np.cumprod(1 + sim_ret)
    peak = np.maximum.accumulate(sim_equity, axis=1)
    drawdown = (sim_equity - peak) / peak
    max_dd = drawdown.min(axis=1)
    avg_dd = max_dd.mean()
    final_equity = sim_equity[:, -1]
    avg_return = final_equity.mean() - 1
    print(f"{regime}: semanas={len(returns)}, retorno esperado 52sem={avg_return:.2%}, drawdown medio={avg_dd:.2%}")
