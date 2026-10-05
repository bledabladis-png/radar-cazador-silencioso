# -*- coding: utf-8 -*-
"""Tests render de PNGs ciegos.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.constants import FIELDS_REQUIRED
from scripts.gold_standard.render import (
    render_case,
    render_all,
    _slice_window,
    WIDTH_PX,
    HEIGHT_PX,
)


def _make_ticker_df(n=500, seed=42, ticker="AAA"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    price = 100 + np.cumsum(rng.normal(0, 1, n))
    high = price + np.abs(rng.normal(0, 0.5, n))
    low = price - np.abs(rng.normal(0, 0.5, n))
    open_ = price + rng.normal(0, 0.3, n)
    close = price
    volume = np.abs(rng.normal(1e6, 2e5, n))
    cols = pd.MultiIndex.from_tuples(
        [(f, ticker) for f in FIELDS_REQUIRED]
    )
    df = pd.DataFrame(
        np.column_stack([open_, high, low, close, volume]),
        index=idx, columns=cols,
    )
    return df


def _make_dataset_from_ticker(ticker_df, ticker="AAA"):
    return ticker_df


def test_slice_window_corta_240(tmp_path):
    df = _make_ticker_df(n=500)
    sub = df.xs("AAA", axis=1, level=1)
    sl = _slice_window(sub, sub.index[-1], window=240)
    assert len(sl) == 240
    assert sl.index[-1] == sub.index[-1]


def test_slice_window_falla_si_insuficiente():
    df = _make_ticker_df(n=100)
    sub = df.xs("AAA", axis=1, level=1)
    with pytest.raises(ValueError, match="Ventana incompleta"):
        _slice_window(sub, sub.index[-1], window=240)


def test_render_case_escribe_png(tmp_path):
    df = _make_ticker_df(n=500)
    sub = df.xs("AAA", axis=1, level=1)
    out = tmp_path / "case_12345678.png"
    render_case(sub, sub.index[-1], 12345678, out)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_render_case_dimensiones(tmp_path):
    from PIL import Image
    df = _make_ticker_df(n=500)
    sub = df.xs("AAA", axis=1, level=1)
    out = tmp_path / "case_1.png"
    render_case(sub, sub.index[-1], 1, out)
    with Image.open(out) as im:
        w, h = im.size
    assert w == WIDTH_PX
    assert h == HEIGHT_PX


def test_render_all_cuenta_correctamente(tmp_path):
    df = _make_ticker_df(n=500)
    sample = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "t": [df.index[-1], df.index[-10]],
        "blind_id": [1, 2],
    })
    res = render_all(sample, df, tmp_path)
    assert res["ok"] == 2
    assert res["fail"] == 0
    assert (tmp_path / "case_1.png").exists()
    assert (tmp_path / "case_2.png").exists()


def test_render_all_registra_fallos(tmp_path):
    df = _make_ticker_df(n=100)
    sample = pd.DataFrame({
        "ticker": ["AAA"],
        "t": [df.index[-1]],
        "blind_id": [1],
    })
    res = render_all(sample, df, tmp_path)
    assert res["ok"] == 0
    assert res["fail"] == 1
    assert len(res["errors"]) == 1


def test_render_case_determinista_bytes(tmp_path):
    """Dos renders del mismo caso producen PNG identicos byte a byte."""
    df = _make_ticker_df(n=500)
    sub = df.xs("AAA", axis=1, level=1)
    out1 = tmp_path / "a.png"
    out2 = tmp_path / "b.png"
    render_case(sub, sub.index[-1], 42, out1)
    render_case(sub, sub.index[-1], 42, out2)
    assert out1.read_bytes() == out2.read_bytes()