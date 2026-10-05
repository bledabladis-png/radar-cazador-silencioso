# -*- coding: utf-8 -*-
"""Tests disjuncion A/B (P0.3, dictamen).

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.sampling_a import build_metadata, sample_a


def _make_episodes(tickers, n_per_ticker=500):
    rows = []
    idx = pd.date_range("2021-01-04", periods=n_per_ticker, freq="B")
    for tk in tickers:
        for d in idx:
            rows.append({"ticker": tk, "t": d})
    return pd.DataFrame(rows)


def test_sample_a_excluye_explicitamente():
    eps = _make_episodes(["AAA", "BBB", "CCC"], n_per_ticker=200)
    sm = {tk: "Tech" for tk in ["AAA", "BBB", "CCC"]}
    meta = build_metadata(eps, sm)
    rng = np.random.default_rng(0)
    flags = pd.Series(
        rng.integers(0, 2, size=len(meta)),
        index=pd.MultiIndex.from_arrays([meta["ticker"], meta["t"]]),
    )
    # Excluir los primeros 20 (ticker, t) de AAA
    excluir = meta[meta["ticker"] == "AAA"].head(20)[["ticker", "t"]].copy()
    out = sample_a(
        meta, flags, n_pos=20, n_neg=20,
        min_gap_sessions=5, seed=1, excluir=excluir,
    )
    key_excl = set(zip(excluir["ticker"], excluir["t"]))
    key_out = set(zip(out["ticker"], out["t"]))
    assert key_excl.isdisjoint(key_out)


def test_sample_a_sin_excluir_funciona_igual():
    eps = _make_episodes(["AAA", "BBB"], n_per_ticker=200)
    sm = {"AAA": "Tech", "BBB": "Tech"}
    meta = build_metadata(eps, sm)
    rng = np.random.default_rng(0)
    flags = pd.Series(
        rng.integers(0, 2, size=len(meta)),
        index=pd.MultiIndex.from_arrays([meta["ticker"], meta["t"]]),
    )
    out = sample_a(meta, flags, n_pos=10, n_neg=10,
                   min_gap_sessions=5, seed=1, excluir=None)
    assert len(out) == 20