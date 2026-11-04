# -*- coding: utf-8 -*-
# Tests de indicators/wyckoff_h20.py (WYCKOFF-5BX-INSTRUMENTACION).
import pandas as pd

from indicators.wyckoff_h20 import (
    CONFIRMED_PHASES,
    DEFAULT_MIN_SESSIONS,
    THRESHOLD_5BX,
    count_confirmed_h20_complete,
)


def _df(rows):
    return pd.DataFrame(rows)


def test_constantes():
    assert DEFAULT_MIN_SESSIONS == 20
    assert THRESHOLD_5BX == 550
    assert "ACCUMULATION" in CONFIRMED_PHASES
    assert "DISTRIBUTION" in CONFIRMED_PHASES
    assert "ACCUMULATION_FORMING" not in CONFIRMED_PHASES
    assert "MARKUP" not in CONFIRMED_PHASES
    assert "MARKDOWN" not in CONFIRMED_PHASES


def test_df_vacio():
    r = count_confirmed_h20_complete(pd.DataFrame())
    assert r["n_confirmed_h20_complete"] == 0
    assert r["n_confirmed_total"] == 0
    assert r["threshold_reached"] is False


def test_df_sin_columnas():
    r = count_confirmed_h20_complete(pd.DataFrame({"x": [1]}))
    assert r["n_confirmed_h20_complete"] == 0


def test_solo_forming_no_cuenta():
    df = _df([{
        "ticker": "AAPL",
        "event_phase": "ACCUMULATION_FORMING",
        "phase_since": "2026-01-01",
    }])
    r = count_confirmed_h20_complete(df, reference_date="2026-06-01")
    assert r["n_confirmed_h20_complete"] == 0
    assert r["n_confirmed_total"] == 0


def test_un_confirmado_con_h20_completo():
    df = _df([{
        "ticker": "AAPL",
        "event_phase": "ACCUMULATION",
        "phase_since": "2026-01-05",
    }])
    r = count_confirmed_h20_complete(df, reference_date="2026-03-01")
    assert r["n_confirmed_h20_complete"] == 1
    assert r["by_phase"]["ACCUMULATION"] == 1


def test_un_confirmado_sin_h20_completo():
    df = _df([{
        "ticker": "AAPL",
        "event_phase": "ACCUMULATION",
        "phase_since": "2026-02-27",
    }])
    r = count_confirmed_h20_complete(df, reference_date="2026-03-01")
    assert r["n_confirmed_h20_complete"] == 0


def test_dos_fases_mismo_ticker():
    df = _df([
        {"ticker": "AAPL", "event_phase": "ACCUMULATION", "phase_since": "2025-01-01"},
        {"ticker": "AAPL", "event_phase": "DISTRIBUTION", "phase_since": "2025-06-01"},
    ])
    r = count_confirmed_h20_complete(df, reference_date="2026-03-01")
    assert r["n_confirmed_h20_complete"] == 2
    assert r["by_phase"]["ACCUMULATION"] == 1
    assert r["by_phase"]["DISTRIBUTION"] == 1


def test_deduplicacion_por_ticker_phase_since():
    df = _df([
        {"ticker": "AAPL", "event_phase": "ACCUMULATION", "phase_since": "2025-01-01"},
        {"ticker": "AAPL", "event_phase": "ACCUMULATION", "phase_since": "2025-01-01"},
        {"ticker": "AAPL", "event_phase": "ACCUMULATION", "phase_since": "2025-01-01"},
    ])
    r = count_confirmed_h20_complete(df, reference_date="2026-03-01")
    assert r["n_confirmed_h20_complete"] == 1


def test_phase_since_none_no_cuenta():
    df = _df([
        {"ticker": "AAPL", "event_phase": "ACCUMULATION", "phase_since": None},
    ])
    r = count_confirmed_h20_complete(df, reference_date="2026-03-01")
    assert r["n_confirmed_h20_complete"] == 0


def test_threshold_reached_con_550():
    rows = []
    for i in range(550):
        rows.append({
            "ticker": "T" + str(i).zfill(4),
            "event_phase": "ACCUMULATION",
            "phase_since": "2025-01-01",
        })
    df = pd.DataFrame(rows)
    r = count_confirmed_h20_complete(df, reference_date="2026-03-01")
    assert r["n_confirmed_h20_complete"] == 550
    assert r["threshold_reached"] is True


def test_reference_date_desde_df_si_no_se_pasa():
    df = _df([
        {"ticker": "AAPL", "event_phase": "ACCUMULATION", "phase_since": "2025-01-01", "date": "2026-03-01"},
    ])
    r = count_confirmed_h20_complete(df)
    assert r["n_confirmed_h20_complete"] == 1
    assert r["reference_date"] == "2026-03-01"
