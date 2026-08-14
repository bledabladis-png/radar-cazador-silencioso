# indicators/index_leaders.py - Selecciona las 5 mejores empresas por WLS
import pandas as pd
import numpy as np
from indicators.wyckoff import wyckoff_score, classify_wyckoff_phase, detect_spring, detect_sos
from src.stock_data_loader import normalize_yahoo_ticker
from src.utils import robust_zscore, get_col
from config.index_tickers import INDEX_CONFIG
from data.providers.router import DataRouter

def compute_stock_metrics_for_index(df_stocks, index_name, stock_list, df_index_data=None):
    results = []
    etf_ticker = INDEX_CONFIG[index_name]['index_ticker']
    router = DataRouter()

    # Obtener precio del índice
    if df_index_data is not None and etf_ticker in df_index_data.columns.get_level_values(1):
        price_index = get_col(df_index_data, etf_ticker, 'Close')
    else:
        idx_df = router.get_market_data([etf_ticker], period='1y')
        price_index = get_col(idx_df, etf_ticker, 'Close')

    for ticker in stock_list:
        try:
            # Si el ticker no está en df_stocks, descargarlo
            if df_stocks is not None and ticker in df_stocks.columns.get_level_values(1):
                close = get_col(df_stocks, ticker, 'Close')
                volume = get_col(df_stocks, ticker, 'Volume')
            else:
                try:
                    single_df = router.get_market_data([ticker], period='5y')
                    if single_df is None or ticker not in single_df.columns.get_level_values(1):
                        continue
                    close = get_col(single_df, ticker, 'Close')
                    volume = get_col(single_df, ticker, 'Volume')
                except Exception:
                    continue
        except:
            continue

        if len(close.dropna()) < 60:
            continue

        common_idx = close.index.intersection(price_index.index)
        if len(common_idx) == 0:
            continue
        rs = close.loc[common_idx] / price_index.loc[common_idx]
        rs_mom = np.log(rs).diff(20).iloc[-1]

        ret = close.pct_change(fill_method=None)
        dollar_vol = close * volume
        flow_raw = ret * dollar_vol
        flow_z = robust_zscore(flow_raw, window=60).iloc[-1]

        try:
            source_df = single_df if 'single_df' in locals() else df_stocks
            wyckoff_sc, _, _, _, _, _, _ = wyckoff_score(source_df, ticker)
            wyckoff_sc = wyckoff_sc.iloc[-1] if not wyckoff_sc.empty else np.nan
        except Exception:
            wyckoff_sc = np.nan
        wyckoff_ph = classify_wyckoff_phase(single_df if 'single_df' in locals() else df_stocks, ticker)

        ret_10 = ret.iloc[-10:]
        persistence_10d = (ret_10 > 0).mean() if len(ret_10) > 0 else 0.5

        wyckoff_series = wyckoff_score(single_df if 'single_df' in locals() else df_stocks, ticker)[0]
        if len(wyckoff_series) >= 10:
            score_median = wyckoff_series.rolling(10).median().iloc[-1]
            score_mad = wyckoff_series.rolling(10).apply(lambda x: np.median(np.abs(x - np.median(x)))).iloc[-1]
            stability = np.tanh(score_median / (score_mad + 1e-9))
        else:
            stability = 0.0

        spring = detect_spring(single_df if 'single_df' in locals() else df_stocks, ticker).iloc[-1]
        sos = detect_sos(single_df if 'single_df' in locals() else df_stocks, ticker).iloc[-1]

        results.append({
            'ticker': ticker,
            'rs': rs.iloc[-1] if not rs.empty else np.nan,
            'rs_mom': rs_mom,
            'flow_z': flow_z,
            'wyckoff_score': wyckoff_sc,
            'wyckoff_phase': wyckoff_ph,
            'persistence_10d': persistence_10d,
            'stability': stability,
            'spring': spring,
            'sos': sos
        })

    return pd.DataFrame(results)

def compute_wls_for_index(df_metrics):
    if df_metrics.empty:
        return df_metrics

    def robust_intra(s):
        median = s.median()
        mad = np.median(np.abs(s - median))
        if mad == 0:
            return pd.Series(0.0, index=s.index)
        return (s - median) / (1.4826 * mad + 1e-9)

    df_metrics['rs_z'] = robust_intra(df_metrics['rs']).clip(-3, 3)
    df_metrics['flow_z_norm'] = robust_intra(df_metrics['flow_z']).clip(-3, 3)
    df_metrics['rws_z'] = robust_intra(df_metrics['wyckoff_score']).clip(-3, 3)
    df_metrics['stab_z'] = robust_intra(df_metrics['stability']).clip(-3, 3)

    df_metrics['wls'] = 0.35*df_metrics['rs_z'] + 0.25*df_metrics['flow_z_norm'] + 0.25*df_metrics['rws_z'] + 0.10*df_metrics['stab_z']
    df_metrics['wls'] *= 1 + 0.05 * np.minimum(df_metrics['persistence_10d'], 1.0)

    return df_metrics.sort_values('wls', ascending=False)

def select_index_leaders(df_market, df_stocks, index_names, df_index_data=None):
    holdings = pd.read_csv('data/index_holdings.csv')
    leaders = {}

    for nombre in index_names:
        config = INDEX_CONFIG[nombre]
        etf = config['etf_ticker']
        max_comp = config['max_companies']

        sub = holdings[holdings['etf'] == etf].sort_values('weight', ascending=False)
        tickers = [normalize_yahoo_ticker(t) for t in sub['ticker'].tolist()[:max_comp]]
        if not tickers:
            continue

        metrics = compute_stock_metrics_for_index(df_stocks, nombre, tickers, df_index_data)
        if metrics.empty:
            continue

        wls_df = compute_wls_for_index(metrics)
        leaders[nombre] = wls_df.head(5)

    return leaders
