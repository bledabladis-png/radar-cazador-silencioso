"""DT3 Fase 0: caracterizacion de compute_darkpool_signals.

Contrato observable (consumido por src/pipeline/market_data.py):
  media_dark_pool, n_tickers_ats, n_tickers_total, z_score, week, status

Caracterizacion (regresion interna):
  state, momentum, percentile, z_windows

Bug latente congelado (se corrige en Fase 1):
  fecha = datetime.now() -> test_fecha_actual_es_now
"""
import json
from pathlib import Path

import pytest

from tests.fixtures._darkpool_setup import setup_and_run


FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN_PATH = FIXTURES / "darkpool_golden.json"
TOL = 1e-9


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f"golden ausente: {GOLDEN_PATH}")
    return json.loads(GOLDEN_PATH.read_bytes().decode("utf-8"))


@pytest.fixture
def res(monkeypatch, tmp_path):
    return setup_and_run(monkeypatch, tmp_path)


def test_resultado_no_es_none(res):
    assert res is not None


def test_status_ok(res, golden):
    assert res["status"] == golden["contract"]["status"]


def test_week_coincide(res, golden):
    assert res["week"] == golden["contract"]["week"]


def test_media_dark_pool(res, golden):
    got = float(res["media_dark_pool"])
    exp = float(golden["contract"]["media_dark_pool"])
    assert abs(got - exp) < TOL, f"{got} vs {exp}"


def test_n_tickers_ats(res, golden):
    assert int(res["n_tickers_ats"]) == int(golden["contract"]["n_tickers_ats"])


def test_n_tickers_total(res, golden):
    assert int(res["n_tickers_total"]) == int(golden["contract"]["n_tickers_total"])


def test_z_score(res, golden):
    got = float(res["z_score"])
    exp = float(golden["contract"]["z_score"])
    assert abs(got - exp) < TOL, f"{got} vs {exp}"


def test_state(res, golden):
    assert res["state"] == golden["characterization"]["state"]


def test_momentum(res, golden):
    got = float(res["momentum"])
    exp = float(golden["characterization"]["momentum"])
    assert abs(got - exp) < TOL


def test_percentile(res, golden):
    got = float(res["percentile"])
    exp = float(golden["characterization"]["percentile"])
    assert abs(got - exp) < TOL


def test_z_windows_estructura(res, golden):
    got = res["z_windows"]
    exp = golden["characterization"]["z_windows"]
    assert set(got.keys()) == set(exp.keys())
    for k in exp:
        if exp[k] is None:
            assert got[k] is None, f"{k}: {got[k]} vs None"
        else:
            assert abs(float(got[k]["z"]) - float(exp[k]["z"])) < TOL
            assert got[k]["state"] == exp[k]["state"]


def test_fecha_es_week_start(res):
    """DT3 Fase 1: fecha deriva del dataset (week_start), no de datetime.now()."""
    assert res["fecha"] == res["week"], f"fecha={res['fecha']} week={res['week']}"
