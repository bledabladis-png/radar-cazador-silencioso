import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from regimes.financial_conditions import compute_liquidity_score
from regimes.volatility_regime import compute_volatility_regime
from regimes.macro_regime import compute_macro_regime
from src.utils import get_col
from src.macro_manual_loader import load_macro_manual
from config.tickers import MARKET_TICKERS
from sklearn.metrics import classification_report, confusion_matrix

START = '2007-01-01'
END = datetime.now().strftime('%Y-%m-%d')

def expected_regime(date):
    try:
        df_exp = pd.read_csv('data/expected_regimes.csv', parse_dates=['start', 'end'])
        for _, row in df_exp.iterrows():
            if row['start'] <= date <= row['end']:
                return row['regime']
    except:
        pass
    return 'MIXED'

def apply_persistence_filter(df_res, min_weeks=4):
    df = df_res.copy()
    df['regime_shift'] = (df['obtained'] != df['obtained'].shift()).cumsum()
    racha_len = df.groupby('regime_shift')['obtained'].transform('count')
    return df[racha_len >= min_weeks].copy()

def run_backtest():
    print("Descargando datos completos (20 años)...")
    tickers = []
    for group in MARKET_TICKERS.values():
        if isinstance(group, dict):
            tickers.extend(group.values())
        elif isinstance(group, list):
            tickers.extend(group)
    tickers = list(set(tickers))
    df_full = yf.download(tickers, period='20y', auto_adjust=True)
    if not isinstance(df_full.columns, pd.MultiIndex):
        df_full.columns = pd.MultiIndex.from_tuples(df_full.columns)

    print("Cargando datos macro...")
    df_macro_all = load_macro_manual()
    if df_macro_all is not None:
        df_macro_all['date'] = pd.to_datetime(df_macro_all['date'])

    dates = pd.date_range(START, END, freq='W-FRI')
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
            liq_score, liq_regime, _ = compute_liquidity_score(df_market)
        except:
            liq_score = None
            liq_regime = 'N/A'
        try:
            vix_ret = get_col(df_market, '^VIX', 'Close').pct_change(fill_method=None)
            vol_score, vol_regime, _ = compute_volatility_regime(vix_ret)
        except:
            vol_score = None
            vol_regime = 'N/A'
        try:
            macro_score, macro_regime, macro_conf, _ = compute_macro_regime(
                df_market, df_macro, liq_score, vol_score, previous_regime
            )
        except:
            macro_regime = 'ERROR'
            macro_score = None
            macro_conf = 0.0

        expected = expected_regime(test_date)
        results.append({
            'date': test_date,
            'expected': expected,
            'obtained': macro_regime,
            'macro_score': macro_score.iloc[-1] if macro_score is not None else np.nan,
            'confidence': macro_conf,
            'liq_regime': liq_regime,
            'vol_regime': vol_regime,
        })
        previous_regime = macro_regime

        if len(results) % 200 == 0:
            print(f"Procesados {len(results)} semanas... última fecha: {test_date.date()}")

    df_res = pd.DataFrame(results)
    df_res.to_csv('outputs/backtest_v2_results.csv', index=False)
    print(f"\nResultados guardados. Total semanas: {len(df_res)}")

    # Métricas de clasificación originales
    y_true = df_res['expected']
    y_pred = df_res['obtained']
    labels = sorted(set(y_true) | set(y_pred))

    print("\nMatriz de confusión original:")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    print("\nInforme de clasificación original:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    print("\n--- RENDIMIENTO DEL SPY BAJO CADA RÉGIMEN OBTENIDO ---")
    spy_hist = df_full[('Close', '^GSPC')].resample('W-FRI').last().pct_change().dropna()
    aligned = df_res.set_index('date').join(spy_hist.rename('spy_return'), how='inner')
    for regime in labels:
        mask = aligned['obtained'] == regime
        if mask.sum() > 1:
            avg_ret = aligned.loc[mask, 'spy_return'].mean()
            std_ret = aligned.loc[mask, 'spy_return'].std()
            sharpe = avg_ret / (std_ret + 1e-9) * np.sqrt(52)
            print(f"  {regime}: semanas={mask.sum()}, retorno semanal medio={avg_ret:.2%}, vol={std_ret:.2%}, Sharpe={sharpe:.2f}")

    # Filtro de persistencia
    df_persist = apply_persistence_filter(df_res, min_weeks=4)
    print(f"\n--- TRAS FILTRO DE PERSISTENCIA (rachas ≥4 semanas) ---")
    print(f"Semanas restantes: {len(df_persist)}")
    if len(df_persist) > 0:
        y_true_f = df_persist['expected']
        y_pred_f = df_persist['obtained']
        print("\nMatriz de confusión (filtrada):")
        cm_f = confusion_matrix(y_true_f, y_pred_f, labels=labels)
        print(pd.DataFrame(cm_f, index=labels, columns=labels))
        print("\nInforme de clasificación (filtrado):")
        print(classification_report(y_true_f, y_pred_f, labels=labels, zero_division=0))

        # Duración media tras filtro
        obtained_f = df_persist['obtained'].values
        durations_f = []
        cur_reg = obtained_f[0]
        count = 1
        for i in range(1, len(obtained_f)):
            if obtained_f[i] == cur_reg:
                count += 1
            else:
                durations_f.append(count)
                cur_reg = obtained_f[i]
                count = 1
        durations_f.append(count)
        avg_dur_f = np.mean(durations_f)
        print(f"Duración media tras filtro: {avg_dur_f:.1f} semanas")

    # Duración media original
    print("\n--- DURACIÓN MEDIA DE REGÍMENES OBTENIDOS (ORIGINAL) ---")
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
    print(f"Duración media: {np.mean(durations):.1f} semanas")

if __name__ == "__main__":
    run_backtest()
