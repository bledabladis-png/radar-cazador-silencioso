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


# ============================================================
# WYCKOFF-PHASE-SEMANTICA (2026-11-03, auditoria externa)
# signal_phase = moda de fases con senal; RANGE fallback.
# ============================================================

def _make_dist_signal(pcts):
    """Construye dist con pcts dict {fase: pct}. count = round(pct/10)."""
    cols = {
        'count_accumulation': int(round(pcts.get('pct_accumulation', 0) / 10)),
        'count_accumulation_forming': int(round(pcts.get('pct_accumulation_forming', 0) / 10)),
        'count_markup': int(round(pcts.get('pct_markup', 0) / 10)),
        'count_range': int(round(pcts.get('pct_range', 0) / 10)),
        'count_distribution': int(round(pcts.get('pct_distribution', 0) / 10)),
        'count_distribution_forming': int(round(pcts.get('pct_distribution_forming', 0) / 10)),
        'count_markdown': int(round(pcts.get('pct_markdown', 0) / 10)),
    }
    d = {'sector': ['XLK']}
    d.update({k: [v] for k, v in cols.items()})
    d.update({k: [pcts.get(k, 0.0)] for k in [
        'pct_accumulation', 'pct_accumulation_forming', 'pct_markup',
        'pct_range', 'pct_distribution', 'pct_distribution_forming',
        'pct_markdown',
    ]})
    return pd.DataFrame(d)


def _make_breadth_flow_ok():
    sector = 'XLK'
    breadth = pd.DataFrame({'sector': [sector], 'pct_above_ema50': [60.0], 'date': [pd.Timestamp.now()]})
    flow = pd.DataFrame({'sector': [sector], 'price_ret_20d': [0.02], 'flow_20d_sum': [1000], 'date': [pd.Timestamp.now()]})
    sr = {'ranking': [(sector, 'Tech', 0.5)]}
    return breadth, flow, sr


def test_signal_phase_excluye_range():
    """RANGE 80% + MARKDOWN 20% -> signal_phase=MARKDOWN (no RANGE)."""
    b, f, sr = _make_breadth_flow_ok()
    dist = _make_dist_signal({'pct_range': 80.0, 'pct_markdown': 20.0})
    df = build_sector_regime_matrix(b, f, sr, dist)
    row = df.iloc[0]
    assert row['wyckoff_phase'] == 'RANGE', 'historico (moda bruta) sigue siendo RANGE'
    assert row['wyckoff_signal_phase'] == 'MARKDOWN', 'signal excluye RANGE'
    assert abs(row['wyckoff_signal_share'] - 20.0) < 0.01
    assert row['wyckoff_signal_n'] == 2


def test_signal_phase_fallback_range_sin_senal():
    """Todo RANGE (sin senal) -> signal_phase=RANGE, share=0, N=0."""
    b, f, sr = _make_breadth_flow_ok()
    dist = _make_dist_signal({'pct_range': 100.0})
    df = build_sector_regime_matrix(b, f, sr, dist)
    row = df.iloc[0]
    assert row['wyckoff_signal_phase'] == 'RANGE'
    assert row['wyckoff_signal_share'] == 0.0
    assert row['wyckoff_signal_n'] == 0


def test_signal_phase_desempate_determinista():
    """Empate MARKUP 20% vs MARKDOWN 20% -> gana MARKDOWN (orden conservador)."""
    b, f, sr = _make_breadth_flow_ok()
    dist = _make_dist_signal({'pct_markup': 20.0, 'pct_markdown': 20.0, 'pct_range': 60.0})
    df = build_sector_regime_matrix(b, f, sr, dist)
    row = df.iloc[0]
    assert row['wyckoff_signal_phase'] == 'MARKDOWN'


def test_signal_phase_forming_compone_pool():
    """DIST_FORMING 20% > MARKDOWN 10% -> signal_phase=DISTRIBUTION_FORMING."""
    b, f, sr = _make_breadth_flow_ok()
    dist = _make_dist_signal({'pct_distribution_forming': 20.0, 'pct_markdown': 10.0, 'pct_range': 70.0})
    df = build_sector_regime_matrix(b, f, sr, dist)
    row = df.iloc[0]
    assert row['wyckoff_signal_phase'] == 'DISTRIBUTION_FORMING'
    assert abs(row['wyckoff_signal_share'] - 20.0) < 0.01
