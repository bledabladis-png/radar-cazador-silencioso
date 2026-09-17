"""DT1 Fase 0 (2026-09-17): caracterizacion de compute_sector_scores.

Congela el contrato observable de regimes.sector_regime.compute_sector_scores:
  - ranking (lista de tuplas (ticker, name, score, wyckoff))
  - regime (str)
  - top3 (list de tickers)

Diagnostico (mismo nivel de tolerancia, no promovido a API):
  - last_scores (dict {sector: score})

NO se caracteriza 'components' (se elimina en DT1 Fase 1).

Datos sinteticos reproducibles via RandomState(seed=42), n_days=300.
Golden: tests/fixtures/sector_regime_golden.json
Generador: tests/fixtures/_gen_sector_regime_golden.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.tickers import MARKET_TICKERS
from regimes.sector_regime import compute_sector_scores


FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN_PATH = FIXTURES / "sector_regime_golden.json"

SEED = 42
N_DAYS = 300
TOL = 1e-9


def _make_synthetic_df(seed=SEED, n=N_DAYS):
    sectors = MARKET_TICKERS["sectors"]
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.RandomState(seed)
    data = {}
    data[("Close", "^GSPC")] = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    for k, s in enumerate(sectors):
        drift = 0.0001 * (k + 1)
        close = 100 * np.cumprod(1 + rng.normal(drift, 0.015, n))
        data[("Close", s)] = close
        data[("Open", s)] = close * 0.998
        data[("High", s)] = close * 1.01
        data[("Low", s)] = close * 0.99
        data[("Volume", s)] = rng.randint(1000000, 5000000, n)
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f"golden ausente: {GOLDEN_PATH}")
    return json.loads(GOLDEN_PATH.read_bytes().decode("utf-8"))


@pytest.fixture(scope="module")
def res():
    return compute_sector_scores(_make_synthetic_df())


def test_resultado_no_es_none(res):
    assert res is not None


def test_regime_coincide_con_golden(res, golden):
    assert res["regime"] == golden["contract"]["regime"]


def test_ranking_coincide_con_golden(res, golden):
    ranking = [list(r) for r in res["ranking"]]
    exp = golden["contract"]["ranking"]
    assert len(ranking) == len(exp), f"n={len(ranking)} vs {len(exp)}"
    for got, want in zip(ranking, exp):
        t_g, n_g, s_g, w_g = got
        t_w, n_w, s_w, w_w = want
        assert t_g == t_w, f"ticker: {t_g} vs {t_w}"
        assert n_g == n_w, f"name: {n_g} vs {n_w}"
        assert abs(float(s_g) - float(s_w)) < TOL, (
            f"{t_g}: score {s_g} vs {s_w}"
        )
        assert w_g == w_w, f"wyckoff: {w_g} vs {w_w}"


def test_top3_coincide_con_golden(res, golden):
    top3 = [str(s) for s in res["last_scores"].head(3).index]
    assert top3 == golden["contract"]["top3"]


def test_last_scores_diagnostico(res, golden):
    exp = golden["diagnostic"]["last_scores"]
    got = {k: float(v) for k, v in res["last_scores"].items()}
    assert set(got.keys()) == set(exp.keys())
    for k in exp:
        assert abs(got[k] - exp[k]) < TOL, f"{k}: {got[k]} vs {exp[k]}"