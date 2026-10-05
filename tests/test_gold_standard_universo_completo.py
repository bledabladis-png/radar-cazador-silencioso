# -*- coding: utf-8 -*-
"""Tests: universo completo con UNKNOWN (dictamen DOC 53).

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.constants import FIELDS_REQUIRED
from scripts.gold_standard.sampling_a import build_metadata
from scripts.gold_standard.sector_map import SECTOR_UNKNOWN, assign_sector
from scripts.gold_standard.sampling_frame import build_sampling_frame


def _make_synth(tickers, n_sessions=500, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n_sessions, freq="B")
    cols = []
    for tk in tickers:
        for f in FIELDS_REQUIRED:
            cols.append((f, tk))
    data = rng.normal(100, 5, size=(n_sessions, len(cols)))
    return pd.DataFrame(
        data, index=idx, columns=pd.MultiIndex.from_tuples(cols)
    )


def test_frame_no_filtra_por_sector():
    """El frame es completo, aunque haya tickers sin sector."""
    df = _make_synth(["AAA", "BBB"])
    sf = build_sampling_frame(df)
    assert sf.diagnostics["filtro_sectorial"] is False
    assert sf.n_frame > 0


def test_build_metadata_unknown_si_falta_sector():
    df = _make_synth(["AAA", "BBB"])
    sf = build_sampling_frame(df)
    smap = {"AAA": "XLK"}  # BBB no tiene sector
    meta = build_metadata(sf.episodes, smap)
    assert "sector" in meta.columns
    unknown = meta[meta["sector"] == SECTOR_UNKNOWN]
    assert len(unknown) > 0
    assert set(unknown["ticker"].unique()) == {"BBB"}


def test_build_metadata_no_descarta_observaciones():
    df = _make_synth(["AAA", "BBB"])
    sf = build_sampling_frame(df)
    smap = {}  # ninguno tiene sector
    meta = build_metadata(sf.episodes, smap)
    assert len(meta) == len(sf.episodes)
    assert (meta["sector"] == SECTOR_UNKNOWN).all()


def test_assign_sector_unknown():
    assert assign_sector("XYZ", {}) == SECTOR_UNKNOWN
    assert assign_sector("AAPL", {"AAPL": "XLK"}) == "XLK"