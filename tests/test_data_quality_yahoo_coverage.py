# -*- coding: utf-8 -*-
"""Regresion H1 (2026-10-17): Yahoo Finance debe reportar cobertura.

El commit 1a4e7ea2 (2026-09-12) cambio `Yahoo Finance` de
`sector_breadth.csv` (coverage_logic='sectorial') a
`market_data.parquet` (coverage_logic=None). La rama parquet escribia
coverage=NaN. Desde ese dia el reporte mostraba N/D para la fuente
principal de precios.

El test fabrica el parquet en tmp_path (no lee data/market_data.parquet
real, que esta gitignored y no existe en CI).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _write_market_parquet(root):
    """Fabricar data/market_data.parquet con los 11 sectores + Close no-NaN."""
    sectors = ['XLK', 'XLY', 'XLI', 'XLF', 'XLB', 'XLE',
               'XLC', 'XLU', 'XLP', 'XLV', 'XLRE']
    cols = pd.MultiIndex.from_product(
        [['Close', 'High', 'Low', 'Open', 'Volume'], sectors],
        names=['Price', 'Ticker'],
    )
    idx = pd.date_range('2026-10-01', periods=5, freq='D')
    # Todos los Close tienen dato el ultimo dia -> cobertura 11/11
    data = np.ones((5, len(cols)))
    df = pd.DataFrame(data, index=idx, columns=cols)
    d = root / 'data'
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / 'market_data.parquet')


def test_yahoo_finance_tiene_coverage_sectorial(tmp_path, monkeypatch):
    """Yahoo Finance reporta n_total=11 y coverage=1.0 con parquet completo."""
    _write_market_parquet(tmp_path)
    monkeypatch.chdir(tmp_path)

    from indicators.data_quality import compute_data_quality
    df = compute_data_quality(reference_date=pd.Timestamp('2026-10-05'))

    yahoo = df[df['source'] == 'Yahoo Finance'].iloc[-1]
    assert yahoo['n_total'] == 11
    assert yahoo['n_valid'] == 11
    assert yahoo['coverage'] == pytest.approx(1.0)


def test_yahoo_finance_coverage_parcial(tmp_path, monkeypatch):
    """Si falta 1 sector con Close, coverage = 10/11."""
    _write_market_parquet(tmp_path)
    # Introducir NaN en XLRE del ultimo dia
    import pandas as pd
    p = tmp_path / 'data' / 'market_data.parquet'
    df = pd.read_parquet(p)
    df.loc[df.index[-1], ('Close', 'XLRE')] = np.nan
    df.to_parquet(p)

    monkeypatch.chdir(tmp_path)
    from indicators.data_quality import compute_data_quality
    df_out = compute_data_quality(reference_date=pd.Timestamp('2026-10-05'))

    yahoo = df_out[df_out['source'] == 'Yahoo Finance'].iloc[-1]
    assert yahoo['n_total'] == 11
    assert yahoo['n_valid'] == 10
    assert yahoo['coverage'] == pytest.approx(10.0 / 11.0)