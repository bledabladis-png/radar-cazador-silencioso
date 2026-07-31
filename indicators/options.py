import numpy as np
import pandas as pd
from datetime import datetime
from data.providers.cboe import CboeProvider
from indicators.options_metrics import (
    institutional_hedge_ratio,
    index_volume_share,
    put_share,
    call_share,
    volume_put_call_ratio,
    oi_put_call_ratio,
    classify_pcr,
    classify_ihr,
)

def robust_zscore(series):
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad == 0:
        return 0.0
    return (series.iloc[-1] - median) / (1.4826 * mad)

def compute_pcr_signals():
    provider = CboeProvider()
    if not provider.is_available():
        return None

    data = provider.get_options_data()
    if not data or 'total_pcr' not in data:
        return None

    # ---------- METRICAS DEL DIA ----------
    ihr = institutional_hedge_ratio(data['index_pcr'], data['equity_pcr'])
    idx_vol_share = index_volume_share(data['index_volume'], data['total_volume'])
    p_share = put_share(data['total_put_volume'], data['total_volume'])
    c_share = call_share(data['total_call_volume'], data['total_volume'])
    vol_pcr = volume_put_call_ratio(data['total_put_volume'], data['total_call_volume'])
    oi_pcr = oi_put_call_ratio(data['total_put_oi'], data['total_call_oi'])

    # ---------- HISTORIAL ----------
    try:
        hist = pd.read_csv('outputs/pcr_history.csv', parse_dates=['date'], index_col='date')
    except:
        hist = pd.DataFrame()

    today = pd.Timestamp(data['date'])

    base_cols = [
        'total_pcr', 'index_pcr', 'equity_pcr', 'etp_pcr', 'vix_pcr', 'spx_pcr',
        'total_call_volume', 'total_put_volume', 'total_volume',
        'total_call_oi', 'total_put_oi', 'total_oi',
        'index_call_volume', 'index_put_volume', 'index_volume',
        'index_call_oi', 'index_put_oi', 'index_oi',
        'equity_call_volume', 'equity_put_volume', 'equity_volume',
        'equity_call_oi', 'equity_put_oi', 'equity_oi',
    ]

    if today not in hist.index:
        new_row = {col: data.get(col) for col in base_cols}
        new_df = pd.DataFrame([new_row], index=[today])
        hist = pd.concat([hist, new_df])
        hist.sort_index(inplace=True)
        hist.to_csv('outputs/pcr_history.csv', index_label='date')

    # ---------- Z-SCORE del PCR Total ----------
    pcr_series = hist['total_pcr'] if 'total_pcr' in hist.columns else pd.Series(dtype=float)
    pcr_ewm = None  # definido para evitar UnboundLocalError
    if len(pcr_series) >= 20:
        pcr_ewm = pcr_series.ewm(span=5).mean()
        window = min(252, len(pcr_ewm))
        z_series = pcr_ewm.rolling(window, min_periods=20).apply(lambda x: robust_zscore(pd.Series(x)), raw=False)
        z = z_series.iloc[-1]
        momentum = z_series.ewm(span=5).mean().iloc[-1]
        percentile = (pcr_ewm.iloc[-window:] < pcr_ewm.iloc[-1]).mean() * 100
        state = classify_pcr(z)
        score = np.tanh(z / 2)
    else:
        z = np.nan
        momentum = np.nan
        percentile = np.nan
        state = "Sin historial suficiente"
        score = np.nan

    return {
        'status': 'OK',
        'total_pcr': data['total_pcr'],
        'pcr_ewm': pcr_ewm.iloc[-1] if pcr_ewm is not None else np.nan,
        'z_score': z,
        'momentum': momentum,
        'percentile': percentile,
        'state': state,
        'score': score,
        'index_pcr': data['index_pcr'],
        'equity_pcr': data['equity_pcr'],
        'etp_pcr': data['etp_pcr'],
        'spx_pcr': data['spx_pcr'],
        'vix_pcr': data['vix_pcr'],
        'ihr': ihr,
        'ihr_state': classify_ihr(ihr),
        'index_volume_share': idx_vol_share,
        'put_share': p_share,
        'call_share': c_share,
        'volume_pcr': vol_pcr,
        'oi_pcr': oi_pcr,
        'last_date': data['date'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
