import pandas as pd
import numpy as np
from indicators.sector_regime_matrix import build_sector_regime_matrix


def _make_dfs(price_ret, pct_above, flow_sum, phase):
    sector = 'XLK'
    breadth = pd.DataFrame({'sector': [sector], 'pct_above_ema50': [pct_above], 'date': [pd.Timestamp.now()]})
    flow = pd.DataFrame({'sector': [sector], 'price_ret_20d': [price_ret], 'flow_20d_sum': [flow_sum], 'date': [pd.Timestamp.now()]})
    sector_results = {'ranking': [(sector, 'Tech', 0.5)]}
    dist = pd.DataFrame({
        'sector': [sector],
        'count_accumulation': [10 if phase in ('ACCUMULATION', 'MARKUP') else 0],
        'count_accumulation_forming': [0],
        'count_markup': [0],
        'count_range': [10 if phase == 'RANGE' else 0],
        'count_distribution': [0],
        'count_distribution_forming': [0],
        'count_markdown': [0],
        'pct_accumulation': [50.0 if phase in ('ACCUMULATION', 'MARKUP') else 0.0],
        'pct_markup': [0.0],
        'pct_range': [50.0 if phase == 'RANGE' else 0.0],
        'pct_distribution': [0.0],
        'pct_markdown': [0.0],
    })
    return breadth, flow, sector_results, dist


def test_tres_positivas_alineacion_positiva():
    '''Bloque 2 auditoria externa: structure_pos retirada.
    positive_conditions ahora cuenta solo 3 dimensiones: precio, breadth, flow.
    '''
    b, f, sr, dist = _make_dfs(0.02, 60.0, 1000, 'MARKUP')
    df = build_sector_regime_matrix(b, f, sr, dist)
    assert df.iloc[0]['regime_reading'] == 'Alineación positiva'
    assert df.iloc[0]['positive_conditions'] == 3
    assert df.iloc[0]['data_complete'] == True


def test_dos_positivas_constructivo():
    '''Precio + breadth positivos, flow negativo -> 2 positivas -> Constructivo.'''
    b, f, sr, dist = _make_dfs(0.02, 60.0, -1000, 'MARKUP')
    df = build_sector_regime_matrix(b, f, sr, dist)
    assert df.iloc[0]['regime_reading'] == 'Constructivo'
    assert df.iloc[0]['positive_conditions'] == 2


def test_missing_data():
    sector = 'XLK'
    breadth = pd.DataFrame({'sector': [sector], 'pct_above_ema50': [np.nan], 'date': [pd.Timestamp.now()]})
    flow = pd.DataFrame({'sector': [sector], 'price_ret_20d': [0.02], 'flow_20d_sum': [1000], 'date': [pd.Timestamp.now()]})
    sr = {'ranking': [(sector, 'Tech', 0.5)]}
    dist_missing = pd.DataFrame({
        'sector': [sector],
        'count_accumulation': [0],
        'count_accumulation_forming': [0],
        'count_markup': [0],
        'count_range': [10],
        'count_distribution': [0],
        'count_distribution_forming': [0],
        'count_markdown': [0],
        'pct_accumulation': [0.0],
        'pct_markup': [0.0],
        'pct_range': [50.0],
        'pct_distribution': [0.0],
        'pct_markdown': [0.0],
    })
    df = build_sector_regime_matrix(breadth, flow, sr, dist_missing)
    assert df.iloc[0]['data_complete'] == False
    assert df.iloc[0]['regime_reading'] == 'N/D'


def test_structure_ratio_columnas_presentes():
    '''Bloque 2: sector_regime_matrix publica structure_ratio + n_estructuras.'''
    b, f, sr, dist = _make_dfs(0.02, 60.0, 1000, 'MARKUP')
    df = build_sector_regime_matrix(b, f, sr, dist)
    assert 'structure_ratio' in df.columns
    assert 'n_estructuras' in df.columns
    assert 'structure_positive' not in df.columns
    # dist tiene 10 en count_accumulation (alcista) y 0 en resto -> ratio 1.0
    assert df.iloc[0]['structure_ratio'] == 1.0
    assert df.iloc[0]['n_estructuras'] == 10


def test_structure_ratio_nan_si_todo_range():
    '''Si toda la estructura es RANGE, structure_ratio = None (N/D).'''
    b, f, sr, dist = _make_dfs(0.02, 60.0, 1000, 'RANGE')
    df = build_sector_regime_matrix(b, f, sr, dist)
    assert pd.isna(df.iloc[0]['structure_ratio'])
    assert df.iloc[0]['n_estructuras'] == 0
