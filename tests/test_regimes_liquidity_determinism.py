"""Regresion P7 (DATETIME_NOW) en _fetch_liquidity_v2_data.

Bug: `endDate` se calculaba con `pd.Timestamp.now().date()`. Re-ejecutar
otro dia con los mismos CSVs daba otro `endDate`, rompiendo el
determinismo dado un snapshot. Auditoria AST 2026-10-28.

Fix: `endDate` derivado de `reference_date` via last_expected_market_date.
Fallback a `now()` solo si reference_date es None (comportamiento previo).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import regimes as reg
from src.market_calendar import last_expected_market_date


def _fake_fed_data():
    idx = pd.DatetimeIndex(["2026-10-23", "2026-10-26", "2026-10-27"])
    return pd.DataFrame({"sofr": [4.5, 4.5, 4.5], "effr": [4.4, 4.4, 4.4]}, index=idx)


class _FakeResp:
    status_code = 200
    def json(self):
        return {"refRates": [
            {"effectiveDate": "2026-10-27", "percentPercentile99": "4.6"},
        ]}


def _run(reference_date, monkeypatch, captured):
    monkeypatch.setattr(
        "data.providers.router.DataRouter.get_fed_data",
        lambda self, columns=None: _fake_fed_data(),
    )
    def _fake_get(url, params=None, timeout=None):
        captured["params"] = params
        return _FakeResp()
    with patch("requests.get", side_effect=_fake_get):
        return reg._fetch_liquidity_v2_data(reference_date=reference_date)


def test_endDate_usa_reference_date(monkeypatch):
    captured = {}
    _run(datetime(2026, 10, 27), monkeypatch, captured)
    assert "params" in captured, "requests.get no fue llamado"
    esperado = last_expected_market_date(datetime(2026, 10, 27)).strftime("%Y-%m-%d")
    assert captured["params"]["endDate"] == esperado


def test_endDate_fallback_now_sin_reference(monkeypatch):
    captured = {}
    _run(None, monkeypatch, captured)
    assert "params" in captured
    hoy = pd.Timestamp.now().date().strftime("%Y-%m-%d")
    assert captured["params"]["endDate"] == hoy


def test_endDate_determinista_mismo_reference(monkeypatch):
    cap1 = {}
    cap2 = {}
    _run(datetime(2026, 10, 27, 3, 17), monkeypatch, cap1)
    _run(datetime(2026, 10, 27, 16, 39), monkeypatch, cap2)
    assert cap1["params"]["endDate"] == cap2["params"]["endDate"]
    assert cap1["params"]["endDate"] == last_expected_market_date(
        datetime(2026, 10, 27)).strftime("%Y-%m-%d")


def test_compute_real_liquidity_propaga_reference_date(monkeypatch):
    capturado = {}
    def _fake_fetch(reference_date=None):
        capturado["reference_date"] = reference_date
        return None, None, None
    monkeypatch.setattr(reg, "_fetch_liquidity_v2_data", _fake_fetch)
    monkeypatch.setattr(reg, "compute_liquidity_score_v2",
                        lambda s, e, c: (None, None, None, None))
    fecha = datetime(2026, 10, 27)
    reg.compute_real_liquidity(reference_date=fecha)
    assert capturado["reference_date"] == fecha


def test_compute_all_regimes_pasa_reference_date(monkeypatch):
    capturado = {}
    def _fake_real_liq(reference_date=None):
        capturado["reference_date"] = reference_date
        return (None, None, None, None)
    monkeypatch.setattr(reg, "compute_real_liquidity", _fake_real_liq)
    monkeypatch.setattr(reg, "compute_financial_conditions",
                        lambda df: (0.0, "NEUTRAL", 0.0))
    monkeypatch.setattr(reg, "compute_volatility_regime",
                        lambda *a, **k: (None, None, None))
    monkeypatch.setattr(reg, "compute_macro_regime",
                        lambda *a, **k: (None, None, None))
    fecha = datetime(2026, 10, 27)
    try:
        reg.compute_all_regimes(None, None, reference_date=fecha)
    except Exception:
        pass
    assert capturado.get("reference_date") == fecha, (
        "compute_all_regimes no propaga reference_date a compute_real_liquidity"
    )
