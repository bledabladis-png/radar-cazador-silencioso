# -*- coding: utf-8 -*-

"""

Momentum de amplitud sectorial v1.0

Calcula deltas de amplitud (EMA20/50/200) y detecta expansión o deterioro.

Consume outputs/history/sector_breadth.csv.

No alimenta motores, scores, pesos ni State Machine.

"""

import pandas as pd

import numpy as np



SECTORS = ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']



def _delta(series, dates, days):

    if len(series) < days + 1:

        return np.nan

    current_date = pd.to_datetime(dates.iloc[-1], errors='coerce')

    past_date = pd.to_datetime(dates.iloc[-days-1], errors='coerce')

    if pd.isna(current_date) or pd.isna(past_date):

        return np.nan

    diff_days = (current_date - past_date).days

    # Tolerancia de días naturales: sesiones + fines de semana + festivos

    max_calendar_days = max(days + 2, int(days * 1.6) + 3)

    if diff_days > max_calendar_days:

        return np.nan

    if pd.isna(series.iloc[-1]) or pd.isna(series.iloc[-days-1]):
        return np.nan

    return series.iloc[-1] - series.iloc[-days-1]



def classify_delta(delta):

    if pd.isna(delta):

        return 'N/D'

    if delta >= 5:

        return 'Expansión fuerte'

    elif delta >= 2:

        return 'Expansión moderada'

    elif delta > -2:

        return 'Estable'

    elif delta > -5:

        return 'Deterioro moderado'

    else:

        return 'Deterioro fuerte'



def compute_sector_breadth_momentum(sector_breadth_csv_path):

    df = pd.read_csv(sector_breadth_csv_path, parse_dates=['date'])

    if df.empty:

        return pd.DataFrame()



    df = df.sort_values(['sector','date'])

    rows = []

    for sector in SECTORS:

        sub = df[df['sector'] == sector].drop_duplicates(subset='date', keep='last').sort_values('date')

        if sub.empty:

            continue



        ema20 = sub['pct_above_ema20']

        ema50 = sub['pct_above_ema50']

        ema200 = sub['pct_above_ema200']

        dates = sub['date']



        d1_20 = _delta(ema20, dates, 1)

        d5_20 = _delta(ema20, dates, 5)

        d20_20 = _delta(ema20, dates, 20)



        d1_50 = _delta(ema50, dates, 1)

        d5_50 = _delta(ema50, dates, 5)

        d20_50 = _delta(ema50, dates, 20)



        d1_200 = _delta(ema200, dates, 1)

        d5_200 = _delta(ema200, dates, 5)

        d20_200 = _delta(ema200, dates, 20)



        # Expansión 5d sobre EMA20: delta5d >= +5 pp y nivel > 60%

        if pd.notna(d5_20) and d5_20 >= 5 and pd.notna(ema20.iloc[-1]) and ema20.iloc[-1] > 60:

            expansion = True

        else:

            expansion = False



        # Deterioro 5d: delta5d <= -5 pp

        if pd.notna(d5_20) and d5_20 <= -5:

            deterioration = True

        else:

            deterioration = False



        rows.append({

            'date': sub['date'].iloc[-1],

            'sector': sector,

            'delta_1d_ema20': d1_20,

            'delta_5d_ema20': d5_20,

            'delta_20d_ema20': d20_20,

            'delta_1d_ema50': d1_50,

            'delta_5d_ema50': d5_50,

            'delta_20d_ema50': d20_50,

            'delta_1d_ema200': d1_200,

            'delta_5d_ema200': d5_200,

            'delta_20d_ema200': d20_200,

            'classification_5d_ema20': classify_delta(d5_20),

            'classification_5d_ema50': classify_delta(d5_50),

            'classification_5d_ema200': classify_delta(d5_200),

            'breadth_expansion_5d': expansion,

            'breadth_deterioration_5d': deterioration,

        })



    return pd.DataFrame(rows)

