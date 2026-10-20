import pandas as pd
import numpy as np
from indicators.evidence_matrix import compute_evidence_matrix

def _make_dfs(sectors, price, breadth, flow, proxy, wyckoff):
    n = len(sectors)
    # C4-code: anadida columna 'date' para que el writer pueda derivar la fecha.
    return (pd.DataFrame({'date': ['2026-09-11']*n, 'sector': sectors, 'pct_above_ema50': breadth, 'n_valid_ema50': [10]*n, 'n_total': [10]*n}),
            pd.DataFrame({'sector': sectors, 'price_ret_20d': price, 'flow_20d_sum': flow, 'n_obs_20d': [20]*n}),
            pd.DataFrame({'sector': sectors, 'flow_median': proxy, 'coverage_flow': [80.0]*n}),
            pd.DataFrame({'sector': sectors, 'pct_accumulation': [40]*n, 'pct_markup': [30]*n, 'pct_distribution': [15]*n, 'pct_markdown': [15]*n, 'coverage_wyckoff': [80.0]*n}))

def test_n_valid_menor_3_insuficiente():
    sectors = ['XLK','XLF','XLV']
    b, fc, conc, wy = _make_dfs(sectors, [np.nan, np.nan, 0.01], [50.0, 60.0, 70.0], [np.nan, np.nan, 1.0], [np.nan, np.nan, np.nan], None)
    df = compute_evidence_matrix(b, conc, fc, wy)
    assert df.iloc[0]['alignment_reading'] == 'EVIDENCIA INSUFICIENTE'
    assert df.iloc[0]['n_sector_evidence_valid'] == 2  # corregido

def test_n_valid_3_positivo():
    sectors = ['XLK']
    b, fc, conc, wy = _make_dfs(sectors, [0.02], [60.0], [1.0], [0.5], None)
    df = compute_evidence_matrix(b, conc, fc, wy)
    assert df.iloc[0]['alignment_reading'] == 'EVIDENCIA PREDOMINANTEMENTE FAVORABLE'
    assert df.iloc[0]['n_sector_evidence_valid'] == 5

def test_nan_no_neutral():
    sectors = ['XLK']
    b, fc, conc, wy = _make_dfs(sectors, [np.nan], [60.0], [1.0], [np.nan], None)
    df = compute_evidence_matrix(b, conc, fc, wy)
    assert df.iloc[0]['n_sector_evidence_valid'] == 3
    assert df.iloc[0]['n_sector_evidence_neutral'] == 0

# ============================================================
# P2.1 (2026-10-13, auditor): contrato de _wyckoff_sign.
# El hallazgo 'Wyckoff=+1 en 11/11 sectores' NO es un default:
# es favorable > unfavorable. Este test fija la regla y evita
# que un futuro cambio la degrade a 'siempre +1'.
# ============================================================
from indicators.evidence_matrix import _wyckoff_sign


def test_wyckoff_sign_favorable_mayor():
    """accumulation + markup > distribution + markdown -> +1."""
    assert _wyckoff_sign({
        'pct_accumulation': 40.0, 'pct_markup': 30.0,
        'pct_distribution': 15.0, 'pct_markdown': 15.0,
    }) == 1


def test_wyckoff_sign_desfavorable_mayor():
    """distribution + markdown > accumulation + markup -> -1."""
    assert _wyckoff_sign({
        'pct_accumulation': 10.0, 'pct_markup': 10.0,
        'pct_distribution': 40.0, 'pct_markdown': 40.0,
    }) == -1


def test_wyckoff_sign_empate_devuelve_cero():
    """favorable == unfavorable -> 0 (no +1 por defecto)."""
    assert _wyckoff_sign({
        'pct_accumulation': 25.0, 'pct_markup': 25.0,
        'pct_distribution': 25.0, 'pct_markdown': 25.0,
    }) == 0


def test_wyckoff_sign_todo_cero_devuelve_cero_no_mas_uno():
    """Todos los porcentajes a 0 (sector sin datos) -> 0.
    Regresion P2.1: evita que un default genere +1 en todos."""
    assert _wyckoff_sign({
        'pct_accumulation': 0.0, 'pct_markup': 0.0,
        'pct_distribution': 0.0, 'pct_markdown': 0.0,
    }) == 0


def test_wyckoff_sign_nan_devuelve_nan():
    """Cualquier NaN en los 4 campos -> NaN (no 0, no +1)."""
    assert pd.isna(_wyckoff_sign({
        'pct_accumulation': np.nan, 'pct_markup': 30.0,
        'pct_distribution': 15.0, 'pct_markdown': 15.0,
    }))
    assert pd.isna(_wyckoff_sign({
        'pct_accumulation': 40.0, 'pct_markup': np.nan,
        'pct_distribution': 15.0, 'pct_markdown': 15.0,
    }))
    assert pd.isna(_wyckoff_sign({
        'pct_accumulation': 40.0, 'pct_markup': 30.0,
        'pct_distribution': np.nan, 'pct_markdown': 15.0,
    }))
    assert pd.isna(_wyckoff_sign({
        'pct_accumulation': 40.0, 'pct_markup': 30.0,
        'pct_distribution': 15.0, 'pct_markdown': np.nan,
    }))


def test_wyckoff_sign_campos_ausentes_devuelve_nan():
    """Dict sin uno de los 4 campos -> NaN (fail-closed)."""
    assert pd.isna(_wyckoff_sign({
        'pct_accumulation': 40.0, 'pct_markup': 30.0,
    }))


# ============================================================
# REP-AUDIT-20261020-REPORT-XLC-EVIDENCE (bloque A)
# Veto por variables materiales: precio y amplitud negativos
# no pueden dar FAVORABLE aunque 3+ votos sean positivos.
# ============================================================

def _df_xlc(price, breadth, primary, proxy, wyckoff_pos):
    sectors = ['XLC']
    b = pd.DataFrame({'date': ['2026-10-19']*len(sectors), 'sector': sectors,
                      'pct_above_ema50': [breadth],
                      'n_valid_ema50': [10], 'n_total': [10]})
    fc = pd.DataFrame({'sector': sectors,
                       'price_ret_20d': [price],
                       'flow_20d_sum': [primary],
                       'n_obs_20d': [20]})
    conc = pd.DataFrame({'sector': sectors,
                         'flow_median': [proxy],
                         'coverage_flow': [80.0]})
    if wyckoff_pos:
        wy = pd.DataFrame({'sector': sectors,
                           'pct_accumulation': [40.0], 'pct_markup': [30.0],
                           'pct_distribution': [15.0], 'pct_markdown': [15.0],
                           'coverage_wyckoff': [80.0]})
    else:
        wy = pd.DataFrame({'sector': sectors,
                           'pct_accumulation': [10.0], 'pct_markup': [10.0],
                           'pct_distribution': [40.0], 'pct_markdown': [40.0],
                           'coverage_wyckoff': [80.0]})
    return compute_evidence_matrix(b, conc, fc, wy)


def test_price_y_breadth_negativos_no_favorable():
    """Caso XLC 2026-10-19: price=-1, breadth=-1, primary=+1,
    proxy=+1, wyckoff=+1. Antes: FAVORABLE (3 pos, 2 neg).
    Con veto: MIXTA."""
    df = _df_xlc(-0.0406, 26.3, 1.0, 0.5, True)
    row = df.iloc[0]
    assert row['price_evidence'] == -1
    assert row['breadth_evidence'] == -1
    assert row['n_sector_evidence_positive'] == 3
    assert row['n_sector_evidence_negative'] == 2
    assert row['alignment_reading'] == 'EVIDENCIA MIXTA'


def test_price_breadth_positivos_siguen_favorables():
    """Regresion: sin veto (price=+1, breadth=+1, 3+ pos) sigue
    siendo FAVORABLE."""
    df = _df_xlc(0.02, 60.0, 1.0, 0.5, True)
    row = df.iloc[0]
    assert row['price_evidence'] == 1
    assert row['breadth_evidence'] == 1
    assert row['alignment_reading'] == 'EVIDENCIA PREDOMINANTEMENTE FAVORABLE'
