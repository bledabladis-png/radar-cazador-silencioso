# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from src.utils import get_col, robust_zscore

def compute_cross_asset_ratios(df_market):
    """
    Calcula los 12 ratios cross-asset institucionales.
    Retorna un diccionario con el ultimo valor, delta 20d (log-retorno) y z-score de cada ratio.
    """
    ratios = {
        'copper_gold': ('HG=F', 'GC=F'),
        'tlt_ief': ('TLT', 'IEF'),  # Duracion larga vs intermedia (NO es pendiente de curva)
        'tip_ief': ('TIP', 'IEF'),  # Inflation-sensitive bonds vs nominal Treasuries (NO es expectativas de inflacion puras)
        'dxy_em': ('DX-Y.NYB', 'EEM'),
        'hyg_lqd': ('HYG', 'LQD'),
        'kre_spy': ('KRE', 'SPY'),
        'sox_spy': ('SMH', 'SPY'),
        'iyt_spy': ('IYT', 'SPY'),
        'xle_spy': ('XLE', 'SPY'),
        'xlu_spy': ('XLU', 'SPY'),
        'xlv_spy': ('XLV', 'SPY'),
        'xlp_spy': ('XLP', 'SPY'),
    }

    result = {}
    for name, (num, den) in ratios.items():
        try:
            num_close = get_col(df_market, num, 'Close')
            den_close = get_col(df_market, den, 'Close')
            ratio_series = num_close / den_close
            result[name] = float(ratio_series.iloc[-1]) if len(ratio_series) > 0 else None
            
            # Delta 20d usando log-retorno (mas robusto para ratios)
            if len(ratio_series) >= 21:
                log_ret = np.log(ratio_series.iloc[-1] / ratio_series.iloc[-21])
                result[f'{name}_delta20'] = float(np.exp(log_ret) - 1)
            else:
                result[f'{name}_delta20'] = None
            
            # Z-score
            try:
                z = robust_zscore(ratio_series, window=60)
                result[f'{name}_zscore'] = float(z.iloc[-1]) if len(z) > 0 and pd.notna(z.iloc[-1]) else None
            except:
                result[f'{name}_zscore'] = None
                
        except (KeyError, IndexError):
            result[name] = None
            result[f'{name}_delta20'] = None
            result[f'{name}_zscore'] = None

    return result


