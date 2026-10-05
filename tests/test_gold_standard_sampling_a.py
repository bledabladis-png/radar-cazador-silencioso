# -*- coding: utf-8 -*-
"""Tests muestreo A.

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

from scripts.gold_standard.sampling_a import (
    assign_periodo,
    build_metadata,
    apply_min_gap,
    sample_a,
)


def _make_episodes(tickers, n_per_ticker=500):
    """Sesiones laborables desde 2021-01-01 (dentro del estudio)."""
    rows = []
    idx = pd.date_range("2021-01-04", periods=n_per_ticker, freq="B")
    for tk in tickers:
        for d in idx:
            rows.append({"ticker": tk, "t": d})
    return pd.DataFrame(rows)


def test_assign_periodo_mapea_anios():
    assert assign_periodo(2021) == "2021-2022"
    assert assign_periodo(2022) == "2021-2022"
    assert assign_periodo(2023) == "2023-2024"
    assert assign_periodo(2025) == "2025-2026"
    with pytest.raises(ValueError):
        assign_periodo(2020)
    with pytest.raises(ValueError):
        assign_periodo(2030)


def test_build_metadata_anade_sector_y_periodo():
    eps = _make_episodes(["AAA", "BBB"], n_per_ticker=10)
    sm = {"AAA": "Tech", "BBB": "Health"}
    meta = build_metadata(eps, sm)
    assert "sector" in meta.columns
    assert "periodo" in meta.columns
    assert set(meta["sector"].unique()) == {"Tech", "Health"}


def test_build_metadata_falla_si_falta_sector():
    eps = _make_episodes(["AAA", "BBB"], n_per_ticker=5)
    sm = {"AAA": "Tech"}
    with pytest.raises(ValueError, match="Tickers sin sector"):
        build_metadata(eps, sm)


def test_apply_min_gap_respeta_separacion():
    eps = _make_episodes(["AAA"], n_per_ticker=500)
    out = apply_min_gap(eps, min_gap_sessions=60, seed=1)
    ts = sorted(out["t"].tolist())
    for a, b in zip(ts, ts[1:]):
        delta = abs((b - a).days)
        assert delta >= 60


def test_apply_min_gap_reduce_cantidad():
    eps = _make_episodes(["AAA"], n_per_ticker=500)
    out = apply_min_gap(eps, min_gap_sessions=120, seed=1)
    assert len(out) < len(eps)
    assert len(out) > 0


def test_apply_min_gap_determinista():
    eps = _make_episodes(["AAA"], n_per_ticker=500)
    out1 = apply_min_gap(eps, min_gap_sessions=60, seed=7)
    out2 = apply_min_gap(eps, min_gap_sessions=60, seed=7)
    pd.testing.assert_frame_equal(out1, out2)


def test_apply_min_gap_separa_por_ticker():
    eps = pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB", "BBB"],
        "t": pd.to_datetime(["2021-01-04", "2021-01-20", "2021-01-04", "2021-01-20"]),
    })
    out = apply_min_gap(eps, min_gap_sessions=20, seed=1)
    assert (out["ticker"] == "AAA").sum() == 1
    assert (out["ticker"] == "BBB").sum() == 1


def test_sample_a_devuelve_200_mas_200():
    eps = _make_episodes(["AAA", "BBB", "CCC", "DDD"], n_per_ticker=400)
    sm = {tk: "Tech" for tk in ["AAA", "BBB", "CCC", "DDD"]}
    meta = build_metadata(eps, sm)
    rng = np.random.default_rng(0)
    flags = pd.Series(
        rng.integers(0, 2, size=len(meta)),
        index=pd.MultiIndex.from_arrays([meta["ticker"], meta["t"]]),
    )
    out = sample_a(
        meta, flags,
        n_pos=50, n_neg=50,
        min_gap_sessions=10,
        seed=0,
    )
    assert len(out) == 100
    assert (out["detect_sow"] == 1).sum() == 50
    assert (out["detect_sow"] == 0).sum() == 50


def test_sample_a_falla_si_no_hay_suficientes():
    eps = _make_episodes(["AAA"], n_per_ticker=300)
    sm = {"AAA": "Tech"}
    meta = build_metadata(eps, sm)
    flags = pd.Series(
        0,
        index=pd.MultiIndex.from_arrays([meta["ticker"], meta["t"]]),
    )
    with pytest.raises(ValueError, match="positivos"):
        sample_a(meta, flags, n_pos=10, n_neg=10,
                 min_gap_sessions=5, seed=0)


def test_sample_a_marca_etiqueta_A():
    eps = _make_episodes(["AAA", "BBB"], n_per_ticker=400)
    sm = {"AAA": "Tech", "BBB": "Health"}
    meta = build_metadata(eps, sm)
    rng = np.random.default_rng(2)
    flags = pd.Series(
        rng.integers(0, 2, size=len(meta)),
        index=pd.MultiIndex.from_arrays([meta["ticker"], meta["t"]]),
    )
    out = sample_a(meta, flags, n_pos=30, n_neg=30,
                   min_gap_sessions=5, seed=2)
    assert (out["muestra"] == "A").all()