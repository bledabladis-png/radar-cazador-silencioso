# tests/test_fu015_dedup.py
"""Tests FU-015 (2026-09-15): clasificar antes de all_data.append.

Cubre:
- Helper _filter_failed_from_batch: 4 casos limite.
- Regresion estructural: el patron bug (append antes de clasificar) no vuelve.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.stock_data_loader import _filter_failed_from_batch


def _make_multiindex_df(tickers):
    """DataFrame MultiIndex (Close/Open/High/Low/Volume) x tickers."""
    dates = pd.date_range(end='2026-09-15', periods=5, freq='D')
    data = {}
    for i, t in enumerate(tickers):
        for field in ('Close', 'Open', 'High', 'Low', 'Volume'):
            data[(field, t)] = np.arange(5, dtype=float) + i
    return pd.DataFrame(data, index=dates)


def test_filter_failed_empty_list():
    """Lista vacia -> df sin cambios."""
    df = _make_multiindex_df(['AAA', 'BBB'])
    out = _filter_failed_from_batch(df, [])
    assert out.equals(df)


def test_filter_failed_no_multiindex():
    """Columnas no-MultiIndex -> df sin cambios (defensivo)."""
    df = pd.DataFrame({'Close': [1, 2, 3], 'Open': [0, 1, 2]})
    out = _filter_failed_from_batch(df, ['AAA'])
    assert out.equals(df)


def test_filter_failed_removes_columns():
    """Ticker fallido -> sus columnas desaparecen; resto intacto."""
    df = _make_multiindex_df(['AAA', 'BBB', 'CCC'])
    out = _filter_failed_from_batch(df, ['BBB'])
    remaining = set(out.columns.get_level_values(1))
    assert remaining == {'AAA', 'CCC'}
    assert out.shape[1] == 10  # 5 fields * 2 tickers


def test_filter_failed_unknown_ticker_noop():
    """Ticker fallido no presente -> df sin cambios."""
    df = _make_multiindex_df(['AAA', 'BBB'])
    out = _filter_failed_from_batch(df, ['ZZZ'])
    assert out.equals(df)


def test_fu015_no_append_before_classify():
    """Regresion estructural: all_data.append(data_batch) NO debe ir
    antes del loop de clasificacion.

    Busca en el fuente que 'failed_in_batch = []' aparece ANTES del
    unico 'all_data.append(data_batch)' (el del batch; retry y cascada
    usan append(data_single) y append(cascade_data)).
    """
    src = Path('src/stock_data_loader.py').read_text(encoding='utf-8-sig')
    idx_failed = src.find('failed_in_batch = []')
    idx_append = src.find('all_data.append(data_batch)')
    assert idx_failed != -1, "falta failed_in_batch = [] en el fuente"
    assert idx_append != -1, "falta all_data.append(data_batch) en el fuente"
    assert idx_failed < idx_append, (
        "FU-015 regresion: all_data.append(data_batch) vuelve a estar "
        "antes de la clasificacion"
    )
