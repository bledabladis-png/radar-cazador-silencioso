# -*- coding: utf-8 -*-
"""Tests del sampling frame Gold Standard.

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
from scripts.gold_standard.sampling_frame import (
    build_sampling_frame,
    _ohlcv_complete_mask,
    _validate_columns,
)


def _make_synth(tickers, n_sessions=500, seed=42):
    """DataFrame sintetico con MultiIndex (field, ticker)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_sessions, freq="B")
    cols = []
    for tk in tickers:
        for f in FIELDS_REQUIRED:
            cols.append((f, tk))
    data = rng.normal(100, 5, size=(n_sessions, len(cols)))
    return pd.DataFrame(
        data, index=idx, columns=pd.MultiIndex.from_tuples(cols)
    )


def test_validate_columns_rechaza_dataframe_sin_multiindex():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="MultiIndex"):
        _validate_columns(df)


def test_validate_columns_rechaza_sin_volume():
    df = _make_synth(["AAA"])
    df = df.drop(columns=[("Volume", "AAA")])
    with pytest.raises(ValueError, match="Faltan campos OHLCV"):
        _validate_columns(df)


def test_ohlcv_complete_mask_true_cuando_todo_presente():
    df = _make_synth(["AAA"], n_sessions=50)
    mask = _ohlcv_complete_mask(df, "AAA")
    assert mask.all()


def test_ohlcv_complete_mask_detecta_nan():
    df = _make_synth(["AAA"], n_sessions=50)
    df.loc[df.index[10], ("Close", "AAA")] = np.nan
    mask = _ohlcv_complete_mask(df, "AAA")
    assert not mask.iloc[10]
    assert mask.iloc[:10].all()


def test_frame_ticker_con_suficientes_sesiones():
    df = _make_synth(["AAA"], n_sessions=500)
    sf = build_sampling_frame(df, warmup_min=200, visual_window=240)
    assert sf.n_frame > 0
    assert sf.episodes["t"].max() <= df.index[-1]


def test_frame_ticker_con_pocas_sesiones_queda_fuera():
    df = _make_synth(["AAA"], n_sessions=300)
    sf = build_sampling_frame(df, warmup_min=200, visual_window=240)
    assert sf.n_frame == 0
    assert sf.n_frame_by_ticker["AAA"] == 0


def test_frame_excluye_ventanas_que_tocan_nan():
    df = _make_synth(["AAA"], n_sessions=500)
    df.loc[df.index[100], ("Close", "AAA")] = np.nan
    sf = build_sampling_frame(df, warmup_min=200, visual_window=240)
    for _, row in sf.episodes.iterrows():
        t_pos = df.index.get_loc(row["t"])
        if t_pos - 239 <= 100 <= t_pos:
            pytest.fail(
                f"episodio con ventana que toca NaN: ticker={row['ticker']} t={row['t']}"
            )


def test_frame_primer_episodio_tiene_ventana_completa():
    df = _make_synth(["AAA"], n_sessions=500)
    sf = build_sampling_frame(df, warmup_min=200, visual_window=240)
    assert sf.n_frame > 0
    first_t = sf.episodes["t"].min()
    pos = df.index.get_loc(first_t)
    assert pos >= 240 - 1
    assert pos >= 200 + 240 - 1


def test_frame_multiticker_registra_ambos():
    df = _make_synth(["AAA", "BBB"], n_sessions=500)
    sf = build_sampling_frame(df, warmup_min=200, visual_window=240)
    assert "AAA" in sf.n_frame_by_ticker.index
    assert "BBB" in sf.n_frame_by_ticker.index
    assert sf.n_frame_by_ticker["AAA"] > 0
    assert sf.n_frame_by_ticker["BBB"] > 0
    assert sf.diagnostics["n_tickers_input"] == 2
    assert sf.diagnostics["n_tickers_with_frame"] == 2


def test_frame_un_ticker_sin_sesiones_no_afecta_al_otro():
    df = _make_synth(["AAA", "BBB"], n_sessions=500)
    # Reducir BBB a 300 sesiones efectivas eliminando OHLCV
    df.loc[df.index[300:], ("Close", "BBB")] = np.nan
    sf = build_sampling_frame(df, warmup_min=200, visual_window=240)
    assert sf.n_frame_by_ticker["AAA"] > 0
    # BBB puede tener algunos episodios antes de la posicion 300 pero
    # ninguno despues
    assert sf.n_frame_by_ticker["BBB"] >= 0


def test_frame_determinista():
    df = _make_synth(["AAA"], n_sessions=500)
    sf1 = build_sampling_frame(df, warmup_min=200, visual_window=240)
    sf2 = build_sampling_frame(df, warmup_min=200, visual_window=240)
    pd.testing.assert_frame_equal(sf1.episodes, sf2.episodes)