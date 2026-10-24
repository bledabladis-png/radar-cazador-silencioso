import pandas as pd
import numpy as np
from indicators.sector_regime_matrix import build_sector_regime_matrix

def _make_dfs(price_ret, pct_above, flow_sum, phase):
    sector = 'XLK'
    breadth = pd.DataFrame({'sector':[sector], 'pct_above_ema50':[pct_above], 'date':[pd.Timestamp.now()]})
    flow = pd.DataFrame({'sector':[sector], 'price_ret_20d':[price_ret], 'flow_20d_sum':[flow_sum], 'date':[pd.Timestamp.now()]})
    sector_results = {'ranking': [(sector, 'Tech', 0.5)]}
    dist = pd.DataFrame({'sector':[sector],
                         'pct_accumulation':[50.0 if phase in ('ACCUMULATION','MARKUP') else 0.0],
                         'pct_markup':[0.0],
                         'pct_range':[50.0 if phase == 'RANGE' else 0.0],
                         'pct_distribution':[0.0],
                         'pct_markdown':[0.0]})
    return breadth, flow, sector_results, dist

def test_all_positive():
    b, f, sr, dist = _make_dfs(0.02, 60.0, 1000, 'MARKUP')
    df = build_sector_regime_matrix(b, f, sr, dist)
    assert df.iloc[0]['regime_reading'] == 'Alineación positiva'
    assert df.iloc[0]['positive_conditions'] == 4
    assert df.iloc[0]['data_complete'] == True

def test_three_positive():
    b, f, sr, dist = _make_dfs(0.02, 60.0, -1000, 'MARKUP')
    df = build_sector_regime_matrix(b, f, sr, dist)
    assert df.iloc[0]['regime_reading'] == 'Constructivo'
    assert df.iloc[0]['positive_conditions'] == 3

def test_missing_data():
    sector = 'XLK'
    breadth = pd.DataFrame({'sector':[sector], 'pct_above_ema50':[np.nan], 'date':[pd.Timestamp.now()]})
    flow = pd.DataFrame({'sector':[sector], 'price_ret_20d':[0.02], 'flow_20d_sum':[1000], 'date':[pd.Timestamp.now()]})
    sr = {'ranking': [(sector, 'Tech', 0.5, 'MARKUP')]}
    dist_missing = pd.DataFrame({'sector':[sector], 'pct_accumulation':[0.0], 'pct_markup':[0.0], 'pct_range':[50.0], 'pct_distribution':[0.0], 'pct_markdown':[0.0]})
    df = build_sector_regime_matrix(breadth, flow, sr, dist_missing)
    assert df.iloc[0]['data_complete'] == False
    assert df.iloc[0]['regime_reading'] == 'N/D'

def test_phase_positive():
    from indicators.sector_regime_matrix import _phase_positive
    assert _phase_positive('ACCUMULATION') == True
    assert _phase_positive('MARKUP') == True
    assert _phase_positive('RANGE') == False
    assert _phase_positive('DISTRIBUTION') == False
    assert _phase_positive('MARKDOWN') == False