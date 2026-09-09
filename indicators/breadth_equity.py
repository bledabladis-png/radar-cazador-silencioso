# -*- coding: utf-8 -*-

"""

indicators/breadth_equity.py -- Advance/Decline sobre acciones lideres (v3.15 corregido)

Usa breadth_core.py para NH/NL con shift(1) y avances/descensos.

"""

import pandas as pd

import numpy as np


from indicators.breadth_core import compute_new_highs_lows, compute_advances_declines, validate_coverage



def compute_advance_decline(df_stocks):

    tickers = [col[1] for col in df_stocks.columns if col[0] == 'Close']

    if not tickers:

        return None



    data = {}

    for t in tickers:

        try:

            if isinstance(df_stocks.columns, pd.MultiIndex):
                try:
                    data[t] = df_stocks.loc[:, ('Close', t)]
                except KeyError:
                    pass
            else:
                col = f'{t}_Close'
                if col in df_stocks.columns:
                    data[t] = df_stocks[col]

        except KeyError:

            pass

    closes = pd.DataFrame(data)



    if closes.empty:

        return None



    expected = len(tickers)

    validate_coverage(closes, expected, "Breadth Equity")
    active_tickers = int(closes.notna().sum(axis=1).iloc[-1])
    if active_tickers < 20:
        return None  # Cobertura temporal insuficiente

    # Avances/descensos desde modulo comun
    advances, declines, unchanged = compute_advances_declines(closes)

    # Si no hay cambios válidos en la última fila, omitir métrica
    if int(advances.iloc[-1]) == 0 and int(declines.iloc[-1]) == 0:
        return None

    ad_net = advances - declines
    ad_line = ad_net.cumsum()



    # NH/NL usando funcion comun (shift(1), Close)

    nh, nl = compute_new_highs_lows(closes)

    nh_nl = nh - nl



    # Breadth Thrust (media de 10 dias del ratio avances/(avances+declines))

    daily_total = advances + declines

    breadth_ratio = advances / daily_total.replace(0, np.nan)

    breadth_thrust = breadth_ratio.rolling(10, min_periods=10).mean()



    return {

        'advances': int(advances.iloc[-1]),

        'declines': int(declines.iloc[-1]),

        'unchanged': int(unchanged.iloc[-1]),

        'ad_net': int(ad_net.iloc[-1]),

        'ad_line': float(ad_line.iloc[-1]),

        'new_highs': int(nh.iloc[-1]),

        'new_lows': int(nl.iloc[-1]),

        'nh_nl': int(nh_nl.iloc[-1]),

        'breadth_thrust': float(breadth_thrust.iloc[-1]) if pd.notna(breadth_thrust.iloc[-1]) else 0.5,

        'total_tickers': expected,

        'active_tickers': int(closes.notna().sum(axis=1).iloc[-1]),

    }