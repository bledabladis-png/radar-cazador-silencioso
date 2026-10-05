# -*- coding: utf-8 -*-
"""Tests min_gap por clase en sample_a.

Hallazgo 2026-10-05 (preflight): min_gap sobre el conjunto elimina
88% de positivos porque los negativos dominan. Fix: por clase.

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


def _make_episodes(tickers, n_per_ticker=400):
    rows = []
    idx = pd.date_range("2021-01-04", periods=n_per_ticker, freq="B")
    for tk in tickers:
        for d in idx:
            rows.append({"ticker": tk, "t": d})
    return pd.DataFrame(rows)


def test_min_gap_pos_sesion_preserva_positivos():
    """Con pos_sesion, min_gap mide sesiones reales del calendario.

    Test calibrado: 40 positivos en 3 tickers de 400 sesiones,
    min_gap=20. Esperado: al menos 20 sobreviven.
    """
    eps = _make_episodes(["AAA", "BBB", "CCC"], n_per_ticker=400)
    sm = {tk: "Tech" for tk in ["AAA", "BBB", "CCC"]}
    # Construir calendar_map: posicion ordinal por ticker
    cal_map = {}
    for tk in ["AAA", "BBB", "CCC"]:
        sub = eps[eps["ticker"] == tk].sort_values("t")
        cal_map[tk] = {t: i for i, t in enumerate(sub["t"])}
    meta = build_metadata(eps, sm, calendar_map=cal_map)
    assert "pos_sesion" in meta.columns
    rng = np.random.default_rng(0)
    flags_arr = np.zeros(len(meta), dtype=int)
    idx_pos = rng.choice(len(meta), size=40, replace=False)
    flags_arr[idx_pos] = 1
    flags = pd.Series(
        flags_arr,
        index=pd.MultiIndex.from_arrays([meta["ticker"], meta["t"]]),
    )
    out = sample_a(
        meta, flags, n_pos=40, n_neg=40,
        min_gap_sessions=20, seed=1, strict=False,
    )
    n_pos_out = int((out["detect_sow"] == 1).sum())
    # Al menos 20 deben sobrevivir
    assert n_pos_out >= 20


def test_min_gap_conjunto_elimina_mas_positivos_que_por_clase():
    """Documenta el bug: aplicar min_gap al conjunto elimina mas
    positivos que aplicar por clase."""
    from scripts.gold_standard.sampling_a import apply_min_gap
    eps = _make_episodes(["AAA", "BBB", "CCC"], n_per_ticker=400)
    sm = {tk: "Tech" for tk in ["AAA", "BBB", "CCC"]}
    meta = build_metadata(eps, sm)
    rng = np.random.default_rng(0)
    flags_arr = np.zeros(len(meta), dtype=int)
    idx_pos = rng.choice(len(meta), size=40, replace=False)
    flags_arr[idx_pos] = 1
    flags = pd.Series(
        flags_arr,
        index=pd.MultiIndex.from_arrays([meta["ticker"], meta["t"]]),
    )
    df = meta.copy()
    df["detect_sow"] = flags.reindex(
        pd.MultiIndex.from_arrays([df["ticker"], df["t"]])
    ).fillna(0).astype(int).to_numpy()

    # Aplicar al conjunto (bug)
    gap_conj = apply_min_gap(df, min_gap_sessions=20, seed=1)
    pos_conj = int((gap_conj["detect_sow"] == 1).sum())

    # Aplicar por clase (fix)
    pos_clase = apply_min_gap(
        df[df["detect_sow"] == 1], min_gap_sessions=20, seed=1,
    )
    n_pos_clase = len(pos_clase)

    # El fix preserva mas o igual
    assert n_pos_clase >= pos_conj


def test_strict_sigue_funcionando():
    eps = _make_episodes(["AAA", "BBB"], n_per_ticker=400)
    sm = {"AAA": "Tech", "BBB": "Tech"}
    meta = build_metadata(eps, sm)
    rng = np.random.default_rng(0)
    flags = pd.Series(
        rng.integers(0, 2, size=len(meta)),
        index=pd.MultiIndex.from_arrays([meta["ticker"], meta["t"]]),
    )
    out = sample_a(meta, flags, n_pos=30, n_neg=30,
                   min_gap_sessions=5, seed=1, strict=True)
    assert (out["detect_sow"] == 1).sum() == 30
    assert (out["detect_sow"] == 0).sum() == 30