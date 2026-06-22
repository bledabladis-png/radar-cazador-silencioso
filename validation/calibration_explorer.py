import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
from regimes.financial_conditions import compute_liquidity_score
from regimes.volatility_regime import compute_volatility_regime
from regimes.macro_regime import compute_macro_regime
from src.utils import get_col
from src.macro_manual_loader import load_macro_manual
from config.tickers import MARKET_TICKERS
import itertools

# ============================================================
# PARÁMETROS DE EXPLORACIÓN (DEFINIDOS POR EL HUMANO)
# ============================================================
UMBRALES_EXPANSION = [0.05, 0.10, 0.15, 0.20]
UMBRALES_RECOVERY = [-0.10, -0.05, 0.00, 0.05]
UMBRALES_LIQUIDITY = [-2.5, -2.0, -1.5]  # umbral de volatilidad para LIQUIDITY CRISIS

print("Cargando datos del backtest...")
tickers = []
for g in MARKET_TICKERS.values():
    if isinstance(g, dict): tickers.extend(g.values())
    elif isinstance(g, list): tickers.extend(g)
tickers = list(set(tickers))
df_full = yf.download(tickers, period='10y', auto_adjust=True)
if not isinstance(df_full.columns, pd.MultiIndex):
    df_full.columns = pd.MultiIndex.from_tuples(df_full.columns)

df_macro_all = load_macro_manual()
if df_macro_all is not None:
    df_macro_all['date'] = pd.to_datetime(df_macro_all['date'])

# Datos de referencia de regímenes esperados (simplificado)
def expected_regime(date):
    try:
        df_exp = pd.read_csv('data/expected_regimes.csv', parse_dates=['start', 'end'])
        for _, row in df_exp.iterrows():
            if row['start'] <= date <= row['end']:
                return row['regime']
    except:
        pass
    return 'MIXED'

# ============================================================
# FUNCIÓN DE EVALUACIÓN PARA UN CONJUNTO DE UMBRALES
# ============================================================
def evaluar_umbrales(u_exp, u_rec, u_liq):
    """Ejecuta un backtest simplificado (cada 4 semanas para rapidez) con los umbrales dados."""
    dates = pd.date_range('2015-01-01', df_full.index[-1], freq='4W')
    results = []
    previous_regime = 'MIXED'

    for test_date in dates:
        df_market = df_full[df_full.index <= test_date].copy()
        if df_market.empty:
            continue
        df_macro = None
        if df_macro_all is not None:
            df_macro = df_macro_all[df_macro_all['date'] <= test_date].copy()

        try:
            liq_score, _, _ = compute_liquidity_score(df_market)
            vix_ret = get_col(df_market, '^VIX', 'Close').pct_change(fill_method=None)
            vol_score, _, _ = compute_volatility_regime(vix_ret)
            
            # Obtenemos las señales sin modificar el archivo original
            from scores.macro_scores import compute_macro_signals, compute_macro_score
            signals = compute_macro_signals(df_market, df_macro, liq_score, vol_score)
            macro_score = compute_macro_score(signals)
            
            last = macro_score.iloc[-1]
            last_ms = signals['market_strength'].iloc[-1] if 'market_strength' in signals.columns else 0
            last_inf = signals['inflation'].iloc[-1] if 'inflation' in signals.columns else 0
            last_liq = signals['liquidity'].iloc[-1] if 'liquidity' in signals.columns else 0
            last_vol = signals['volatility'].iloc[-1] if 'volatility' in signals.columns else 0
            last_curve = signals['curve'].iloc[-1] if 'curve' in signals.columns else 0
            last_credit = signals['credit'].iloc[-1] if 'credit' in signals.columns else 0

            # Volatilidad percentil (simplificado)
            vol_pct = 0.5
            if 'volatility' in signals.columns:
                vol_hist = signals['volatility'].dropna()
                vol_pct = (vol_hist < last_vol).mean() if len(vol_hist) > 200 else 0.5

            # CLASIFICACIÓN CON LOS UMBRALES PROBADOS
            regime = 'MIXED'
            if last_vol < u_liq or (last_vol < -1.5 and vol_pct < 0.05):
                regime = 'LIQUIDITY CRISIS'
            elif last_vol < -2.0 and last < 0.1:
                regime = 'LIQUIDITY CRISIS'
            elif last_vol < -1.5 and last_credit < -0.5:
                regime = 'LIQUIDITY CRISIS'
            elif last < -0.4 and last_ms < -0.5:
                regime = 'RECESSION'
            elif last_inf > 0.3 and last_ms < 0:
                regime = 'INFLATION SHOCK'
            elif last < -0.2 and last_inf > 0.3:
                regime = 'STAGFLATION'
            elif last > u_exp and last_ms > 0 and last_liq > -0.1 and last_vol < 0 and last_curve > 0:
                regime = 'EXPANSION'
            elif last > u_rec and last_ms > 0:
                regime = 'RECOVERY'
            elif last > 0.2 and last_ms > 0.2 and last_inf < 0 and last_liq > 0 and last_vol < 0:
                regime = 'GOLDILOCKS'
            elif last < -0.2 and last_ms < 0:
                regime = 'SLOWDOWN'
            elif last > 0 and last_inf < -0.5:
                regime = 'DEFLATION'

            expected = expected_regime(test_date)
            results.append({'date': test_date, 'expected': expected, 'obtained': regime})
        except:
            pass

    if len(results) < 50:
        return None

    df_res = pd.DataFrame(results)
    
    # Métricas
    accuracy = (df_res['expected'] == df_res['obtained']).mean()
    
    # Rendimientos del SPY por régimen
    spy_hist = df_full[('Close', '^GSPC')].resample('W').last().pct_change().dropna()
    aligned = df_res.set_index('date').join(spy_hist.rename('spy_return'), how='inner')
    
    stress_regimes = ['LIQUIDITY CRISIS', 'RECESSION', 'SLOWDOWN']
    positive_regimes = ['EXPANSION', 'RECOVERY']
    
    ret_stress = aligned[aligned['obtained'].isin(stress_regimes)]['spy_return']
    ret_pos = aligned[aligned['obtained'].isin(positive_regimes)]['spy_return']
    
    sharpe_stress = (ret_stress.mean() / (ret_stress.std() + 1e-9)) * np.sqrt(52) if len(ret_stress) > 1 else np.nan
    sharpe_pos = (ret_pos.mean() / (ret_pos.std() + 1e-9)) * np.sqrt(52) if len(ret_pos) > 1 else np.nan
    
    # Separación (diferencia entre Sharpe positivo y de estrés)
    separacion = sharpe_pos - sharpe_stress if pd.notna(sharpe_pos) and pd.notna(sharpe_stress) else np.nan
    
    # Número de regímenes distintos (diversidad)
    n_regimes = df_res['obtained'].nunique()
    
    # Duración media
    obtained = df_res['obtained'].values
    durations = []
    cur_reg = obtained[0]
    count = 1
    for i in range(1, len(obtained)):
        if obtained[i] == cur_reg:
            count += 1
        else:
            durations.append(count)
            cur_reg = obtained[i]
            count = 1
    durations.append(count)
    duracion_media = np.mean(durations) if durations else 0

    return {
        'accuracy': accuracy,
        'sharpe_stress': sharpe_stress,
        'sharpe_positivo': sharpe_pos,
        'separacion': separacion,
        'n_regimes': n_regimes,
        'duracion_media': duracion_media
    }

# ============================================================
# EXPLORACIÓN
# ============================================================
print("\nExplorando combinaciones de umbrales...")
resultados = []
for u_exp, u_rec, u_liq in itertools.product(UMBRALES_EXPANSION, UMBRALES_RECOVERY, UMBRALES_LIQUIDITY):
    print(f"  Probando: EXP={u_exp}, REC={u_rec}, LIQ={u_liq}")
    metrica = evaluar_umbrales(u_exp, u_rec, u_liq)
    if metrica is not None:
        resultados.append({
            'umbral_expansion': u_exp,
            'umbral_recovery': u_rec,
            'umbral_liquidity': u_liq,
            **metrica
        })

df_r = pd.DataFrame(resultados).sort_values('separacion', ascending=False)

print("\n" + "=" * 100)
print("RESULTADOS DE LA EXPLORACIÓN DE UMBRALES (ordenados por separación de Sharpe)")
print("=" * 100)
print(f"\n{'EXP':<6} {'REC':<6} {'LIQ':<6} {'Prec':<8} {'SharpeStr':<10} {'SharpePos':<10} {'Separa':<8} {'NReg':<5} {'DurMed':<8}")
print("-" * 100)
for _, row in df_r.head(20).iterrows():
    print(f"{row['umbral_expansion']:<6.2f} {row['umbral_recovery']:<6.2f} {row['umbral_liquidity']:<6.1f} "
          f"{row['accuracy']:<8.1%} {row['sharpe_stress']:<10.2f} {row['sharpe_positivo']:<10.2f} "
          f"{row['separacion']:<8.2f} {row['n_regimes']:<5} {row['duracion_media']:<8.1f}")

print("\n" + "=" * 100)
print("GUÍA DE INTERPRETACIÓN:")
print("  - Separación alta → El modelo discrimina bien entre entornos positivos y negativos.")
print("  - NReg > 4 → El modelo usa varios regímenes (no solo MIXED).")
print("  - DurMed entre 3 y 15 → Regímenes ni demasiado breves ni eternos.")
print("  - Precisión: orientativa (depende del benchmark manual).")
print("\nTÚ DEBES ELEGIR LA COMBINACIÓN QUE CONSIDERES MÁS ADECUADA.")
print("=" * 100)
