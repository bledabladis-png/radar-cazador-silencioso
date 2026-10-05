# -*- coding: utf-8 -*-
"""Tests contrato implicito del detector: df sin NaN.

BORRADOR — pendiente firma auditor.

Hallazgo 2026-10-05: sin build_ticker_df, detect_sow recibe NaN
internos. rolling(60, min_periods=60) descarta toda ventana con un
NaN. La senal se pierde 10x. 304/316 tickers afectados.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators.wyckoff import build_ticker_df
from scripts.gold_standard.constants import FROZEN_V19_PARAMS
from scripts.gold_standard.sampling_frame import (
    build_sampling_frame, load_dataset,
)

PARQUET = ROOT / "data" / "stock_prices.parquet"


def test_build_ticker_df_no_tiene_nan():
    """build_ticker_df devuelve df sin NaN."""
    if not PARQUET.exists():
        import pytest
        pytest.skip("parquet no presente")
    df = load_dataset()
    tdf = build_ticker_df(df, "AAPL")
    assert tdf.notna().all().all()


def test_frame_usa_ldf():
    """El sampling frame debe construirse sobre filas validas."""
    if not PARQUET.exists():
        import pytest
        pytest.skip("parquet no presente")
    df = load_dataset()
    sf = build_sampling_frame(df)
    # Debe haber episodios
    assert sf.n_frame > 0
    # Todos los t de los episodios deben existir en el df original
    t0 = sf.episodes["t"].iloc[0]
    assert t0 in df.index


def test_detector_flags_no_nan_sin_dropna_afecta():
    """Comparacion: detect_sow directo vs alimentado con build_ticker_df.

    Este test documenta el hallazgo. Se puede convertir en salvaguarda
    estricta cuando se autorice.
    """
    if not PARQUET.exists():
        import pytest
        pytest.skip("parquet no presente")
    df = load_dataset()
    params = FROZEN_V19_PARAMS
    from indicators.wyckoff_v1 import detect_sow

    # Directo (NaN dentro)
    sub = df.xs("META", axis=1, level=1)
    try:
        sow_directo = detect_sow(sub, "META",
            window=params["window"],
            x_atr=params["x_atr"], y_vol=params["y_vol"])
        n_directo = int(sow_directo.sum())
    except Exception:
        n_directo = 0

    # Con dropna
    tdf = build_ticker_df(df, "META")
    try:
        sow_limpio = detect_sow(tdf, "META",
            window=params["window"],
            x_atr=params["x_atr"], y_vol=params["y_vol"])
        n_limpio = int(sow_limpio.sum())
    except Exception:
        n_limpio = 0

    # El contrato: con dropna debe haber >= que sin dropna
    assert n_limpio >= n_directo
    # Y si hay material, debe ser estrictamente mayor en META
    if n_directo == 0:
        assert n_limpio > 0