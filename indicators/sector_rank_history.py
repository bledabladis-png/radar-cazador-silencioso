# -*- coding: utf-8 -*-
"""
Rotación sectorial histórica reciente v1.0
Guarda el ranking sectorial diario y calcula cambios de posición (ΔRank).
No alimenta motores, scores, pesos ni State Machine.
"""
import pandas as pd
import numpy as np
from pathlib import Path

SECTORS = ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']

def update_rank_history(sector_results, history_csv_path, date=None):
    if sector_results is None or 'ranking' not in sector_results:
        return None, None

    ranking = sector_results['ranking']
    if not ranking:
        return None, None

    if date is None:
        date = pd.Timestamp.now().normalize()

    current = []
    for i, (ticker, name, score, fase) in enumerate(ranking[:len(SECTORS)], 1):
        if ticker in SECTORS:
            current.append({'date': date, 'sector': ticker, 'score': score, 'rank': i})
    current_df = pd.DataFrame(current)

    history_path = Path(history_csv_path)
    if history_path.exists():
        try:
            hist = pd.read_csv(history_path, parse_dates=['date'])
        except Exception:
            hist = pd.DataFrame(columns=['date','sector','score','rank'])
    else:
        hist = pd.DataFrame(columns=['date','sector','score','rank'])

    combined = pd.concat([hist, current_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['date','sector'], keep='last')
    combined = combined.sort_values(['date','sector']).reset_index(drop=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(history_path, index=False)

    latest_date = combined['date'].max()
    current_date_data = combined[combined['date'] == latest_date].set_index('sector')
    deltas = []

    for sector in SECTORS:
        if sector not in current_date_data.index:
            continue
        current_rank = current_date_data.loc[sector, 'rank']
        sector_hist = combined[combined['sector'] == sector].set_index('date')['rank'].sort_index()

        def get_delta(days):
            if len(sector_hist) < days + 1:
                return np.nan
            recent = sector_hist.iloc[-days-1:]
            return current_rank - recent.iloc[0]

        delta5 = get_delta(5)
        delta10 = get_delta(10)
        delta20 = get_delta(20)

        def lectura(delta):
            if pd.isna(delta):
                return 'N/D'
            if delta <= -3:
                return 'Fuerte mejora'
            elif -2 <= delta <= 2:
                return 'Estable'
            else:
                return 'Fuerte deterioro'

        deltas.append({
            'sector': sector,
            'rank_actual': current_rank,
            'rank_change_5d': delta5,
            'rank_change_10d': delta10,
            'rank_change_20d': delta20,
            'lectura_5d': lectura(delta5),
            'lectura_10d': lectura(delta10),
            'lectura_20d': lectura(delta20),
        })

    return combined, pd.DataFrame(deltas)
