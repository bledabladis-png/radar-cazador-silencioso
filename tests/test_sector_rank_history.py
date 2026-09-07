import pandas as pd
import numpy as np
from indicators.sector_rank_history import update_rank_history

def _make_results(rankings_dict):
    # rankings_dict: fecha -> lista de tuplas ordenadas
    return {pd.Timestamp(k): v for k, v in rankings_dict.items()}

def test_update_delta5_with_six_dates(tmp_path):
    hist_path = str(tmp_path / 'rank.csv')
    base_ranking = [
        ('XLK',0.5),('XLF',0.4),('XLV',0.3),('XLE',0.2),('XLY',0.1),
        ('XLP',0.0),('XLI',-0.1),('XLB',-0.2),('XLU',-0.3),('XLRE',-0.4),('XLC',0.34)
    ]
    changed_ranking = [
        ('XLC',0.9),('XLE',0.8),('XLV',0.7),('XLF',0.6),('XLK',0.5),
        ('XLP',0.2),('XLY',0.1),('XLI',0.0),('XLB',-0.1),('XLRE',-0.2),('XLU',-0.3)
    ]
    dates = pd.date_range('2026-08-01', periods=6, freq='D')
    # Para las primeras 5 fechas usamos base_ranking, en la última usamos changed
    for i, d in enumerate(dates):
        rank = changed_ranking if i == 5 else base_ranking
        res = {'ranking': [(t, '', s, '') for t, s in rank]}
        _, _ = update_rank_history(res, hist_path, date=d)

    # Ahora leemos el CSV generado y ejecutamos update con la última fecha nuevamente para obtener deltas
    res_last = {'ranking': [(t, '', s, '') for t, s in changed_ranking]}
    _, deltas = update_rank_history(res_last, hist_path, date=dates[-1])
    xlc = deltas[deltas['sector']=='XLC'].iloc[0]
    # XLC pasó de rank 11 a rank 1; pero solo hay 6 fechas, en base era rank 11
    assert xlc['rank_change_5d'] == -10
    assert xlc['lectura_5d'] == 'Fuerte mejora'

def test_insufficient_history_returns_nan(tmp_path):
    hist_path = str(tmp_path / 'rank2.csv')
    ranking = [(t, s) for t, s in [('XLK',0.5),('XLF',0.4),('XLV',0.3),('XLE',0.2),('XLY',0.1),('XLP',0.0),('XLI',-0.1),('XLB',-0.2),('XLU',-0.3),('XLRE',-0.4),('XLC',0.34)]]
    for d in pd.date_range('2026-08-01', periods=2, freq='D'):
        res = {'ranking': [(t, '', s, '') for t, s in ranking]}
        update_rank_history(res, hist_path, date=d)
    res_last = {'ranking': [(t, '', s, '') for t, s in ranking]}
    _, deltas = update_rank_history(res_last, hist_path, date=pd.Timestamp('2026-08-02'))
    xlc = deltas[deltas['sector']=='XLC'].iloc[0]
    assert pd.isna(xlc['rank_change_5d'])
    assert xlc['lectura_5d'] == 'N/D'