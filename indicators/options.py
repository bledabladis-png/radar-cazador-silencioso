import numpy as np

import os
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



def _zscore_last_in_window(series):
    """Z-score robusto de la ULTIMA observacion de la ventana.

    A3.4-03 (2026-09-28): renombrada de robust_zscore -> _zscore_last_in_window.
    Motivo: contrato DISTINTO al canonico src/utils.robust_zscore.
      - canonico: recibe serie completa, rolling(window=60), devuelve Series.
      - esta:     recibe ventana ya recortada, calcula mediana/MAD estatica
                  sobre la ventana, devuelve ESCALAR (el z-score de la ultima
                  observacion).
    Uso unico: options.py:138 dentro de rolling.apply, que le pasa la ventana
    ya recortada. El nombre anterior (robust_zscore) sugeria equivalencia con
    el canonico; no la hay. Ver AUDITORIA_CONSOLIDADA_2026-09-26.md F5.6-15.
    """
    median = series.median()
    # 2026-10-02: np.nanmedian + guard pd.isna(mad). Con NaN en la ventana
    # (huecos de PCR, festivos), np.median propaga NaN silenciosamente a
    # todo el z-score. Mismo patron que el bug arreglado en index_leaders
    # (2f956ed, robust_intra).
    mad = np.nanmedian(np.abs(series - median))
    if pd.isna(mad) or mad == 0:
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
        hist = pd.read_csv('outputs/history/pcr_history.csv', parse_dates=['date'], index_col='date')
    except (FileNotFoundError, OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
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

        # Fix 2026-09-29: escritura atomica. Es historico: leido por
        # data_quality.py:91 (frescura) y options.py:89 (autoread).
        # Mismo patron que darkpool_history, sector_rank_history y
        # state_transition.
        _tmp = 'outputs/history/pcr_history.csv.tmp'
        hist.to_csv(_tmp, index_label='date')
        os.replace(_tmp, 'outputs/history/pcr_history.csv')



    # ---------- Z-SCORE del PCR Total ----------

    pcr_series = hist['total_pcr'] if 'total_pcr' in hist.columns else pd.Series(dtype=float)

    pcr_ewm = None  # definido para evitar UnboundLocalError

    if len(pcr_series) >= 20:

        pcr_ewm = pcr_series.ewm(span=5).mean()

        window = min(252, len(pcr_ewm))

        z_series = pcr_ewm.rolling(window, min_periods=20).apply(lambda x: _zscore_last_in_window(pd.Series(x)), raw=False)

        z = z_series.iloc[-1]

        momentum = z_series.ewm(span=5).mean().iloc[-1]

        percentile = (pcr_ewm.iloc[-window:] < pcr_ewm.iloc[-1]).mean() * 100
        if len(pcr_ewm) >= 15:
            percentile_20d = (pcr_ewm.iloc[-20:] < pcr_ewm.iloc[-1]).mean() * 100
        else:
            percentile_20d = np.nan
        state = classify_pcr(z)

        score = np.tanh(z / 2)

    else:

        z = np.nan

        momentum = np.nan

        percentile = np.nan

        state = "Sin historial suficiente"

        score = np.nan

        percentile = np.nan
        percentile_20d = np.nan
    return {

        'status': 'OK',

        'total_pcr': data['total_pcr'],

        'pcr_ewm': pcr_ewm.iloc[-1] if pcr_ewm is not None else np.nan,

        'z_score': z,

        'momentum': momentum,

        'percentile': percentile,

        'state': state,

        'score': score,

        'percentile': percentile,
        'percentile_20d': percentile_20d,
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

