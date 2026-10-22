# -*- coding: utf-8 -*-
"""Test: cboe_vix3m.parquet no debe tener fechas no bursatiles.

Causa: CBOE publica filas cuando NYSE esta cerrado (observacion del
New Year en viernes: 2010-12-31, 2021-12-31). El parquet trackeado
en git no debe contener esas filas.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.market_calendar import is_market_day


PARQUET = Path('data/cboe_vix3m.parquet')
MANIFEST = Path('data/cboe_vix3m.parquet.manifest.json')


def test_no_non_trading_days():
    """Ninguna fila del parquet debe caer en fecha no bursatil NYSE."""
    if not PARQUET.exists():
        pytest.skip('cboe_vix3m.parquet no existe')
    df = pd.read_parquet(PARQUET)
    assert isinstance(df.index, pd.DatetimeIndex)
    non_trading = [d for d in df.index if not is_market_day(d.date())]
    assert non_trading == [], (
        f'{len(non_trading)} fechas no bursatiles: {non_trading[:5]}'
    )


def test_manifest_rows_match_parquet():
    """manifest.content.rows debe coincidir con len(df)."""
    if not PARQUET.exists() or not MANIFEST.exists():
        pytest.skip('artefactos no existen')
    df = pd.read_parquet(PARQUET)
    m = json.loads(MANIFEST.read_text(encoding='utf-8'))
    assert m['content']['rows'] == len(df), (
        f'manifest rows={m["content"]["rows"]} != parquet rows={len(df)}'
    )


def test_last_date_is_expected_session():
    """La fecha mas reciente debe ser sesion NYSE."""
    if not PARQUET.exists():
        pytest.skip('cboe_vix3m.parquet no existe')
    df = pd.read_parquet(PARQUET)
    last = df.index.max()
    assert is_market_day(last.date()), f'ultima fecha {last.date()} no es sesion'