# -*- coding: utf-8 -*-
"""Tests FU-002-bymarket: cobertura por mercado.

Cubre las 3 correcciones del dictamen auditor:
  C-1: last_closed_session por calendario + FU-018, no por datos.
  C-2: (en guard, no aqui) no exencion global INVALID.
  C-3: coverage_at_session es el dato contractual.

Mas punto 5 (UNKNOWN con n>0 -> INVALID) y punto 6 (universo dinamico).
"""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.utils import _compute_by_market, _latest_closed_session


MADRID = ZoneInfo("Europe/Madrid")


def _make_df_multi_market(markets_tickers, n_rows=10, end_date="2026-09-24"):
    """market_tickers: dict {market_name: [tickers]}.

    Acepta cualquier nombre de market; get_market se mockea en los tests.
    """
    idx = pd.date_range(end=end_date, periods=n_rows, freq="B")
    data = {}
    for market, tickers in markets_tickers.items():
        for t in tickers:
            data[("Close", t)] = [100.0 + i for i in range(n_rows)]
    return pd.DataFrame(data, index=idx)


# ---------- _latest_closed_session ----------
def test_latest_closed_session_usa_durante_sesion_devuelve_ayer(monkeypatch):
    """USA abierto a las 15:59 UTC -> ultima sesion cerrada es ayer."""
    from src import market_hours as mh
    # Mock: sesion de hoy NO cerrada a las 15:59 UTC, ayer si
    def fake_is_closed(market, session_date, ref):
        return session_date < ref.date()
    monkeypatch.setattr(mh, "is_session_closed", fake_is_closed)
    monkeypatch.setattr(mh, "is_trading_session", lambda m, d: d.weekday() < 5)

    ref = datetime(2026, 9, 24, 15, 59, tzinfo=ZoneInfo("UTC"))
    result = _latest_closed_session("US_EQUITY", ref)
    assert result is not None
    assert result.strftime("%Y-%m-%d") == "2026-09-23"


def test_latest_closed_session_europa_tras_cierre_devuelve_hoy(monkeypatch):
    """Europa cerrada a las 15:59 UTC -> ultima sesion cerrada es hoy."""
    from src import market_hours as mh
    # Mock: todo lo que sea dia de negociacion esta cerrado
    monkeypatch.setattr(mh, "is_session_closed",
                        lambda market, sd, ref: sd.weekday() < 5)
    monkeypatch.setattr(mh, "is_trading_session", lambda m, d: d.weekday() < 5)

    ref = datetime(2026, 9, 24, 15, 59, tzinfo=ZoneInfo("UTC"))
    result = _latest_closed_session("EURONEXT", ref)
    assert result is not None
    assert result.strftime("%Y-%m-%d") == "2026-09-24"


def test_latest_closed_session_weekend_retrocede_a_viernes(monkeypatch):
    """Si reference_date es sabado, la ultima sesion cerrada es viernes."""
    from src import market_hours as mh
    monkeypatch.setattr(mh, "is_session_closed",
                        lambda market, sd, ref: sd.weekday() < 5)
    monkeypatch.setattr(mh, "is_trading_session", lambda m, d: d.weekday() < 5)

    # Sabado 2026-09-26
    ref = datetime(2026, 9, 26, 12, 0, tzinfo=ZoneInfo("UTC"))
    result = _latest_closed_session("US_EQUITY", ref)
    assert result is not None
    assert result.strftime("%Y-%m-%d") == "2026-09-25"  # viernes


def test_latest_closed_session_tz_naive_devuelve_none():
    ref = datetime(2026, 9, 24, 15, 59)  # sin tz
    assert _latest_closed_session("US_EQUITY", ref) is None


# ---------- _compute_by_market ----------
def test_bymarket_estructura_completa(monkeypatch):
    """Devuelve entradas para todos los mercados de _KNOWN_MARKETS."""
    from src import instrument_registry as ir
    from src import market_hours as mh
    from src.market_hours import _KNOWN_MARKETS

    monkeypatch.setattr(ir, "get_market", lambda t: "US_EQUITY")
    monkeypatch.setattr(mh, "is_session_closed",
                        lambda market, sd, ref: sd.weekday() < 5)
    monkeypatch.setattr(mh, "is_trading_session", lambda m, d: d.weekday() < 5)

    df = _make_df_multi_market({"US_EQUITY": ["AAPL", "MSFT", "GOOG"]})
    ref = datetime(2026, 9, 24, 15, 59, tzinfo=ZoneInfo("UTC"))
    result = _compute_by_market(df, ref)

    for m in _KNOWN_MARKETS:
        assert m in result
    assert result["US_EQUITY"]["n"] == 3
    assert result["US_EQUITY"]["status"] == "VALID"


def test_bymarket_c1_no_enmascara_ausencia_completa(monkeypatch):
    """C-1: si la sesion esperada no tiene datos, INVALID (no retrocede)."""
    from src import instrument_registry as ir
    from src import market_hours as mh

    monkeypatch.setattr(ir, "get_market", lambda t: "US_EQUITY")
    monkeypatch.setattr(mh, "is_session_closed",
                        lambda market, sd, ref: sd.weekday() < 5)
    monkeypatch.setattr(mh, "is_trading_session", lambda m, d: d.weekday() < 5)

    # df con ultima fecha el 23, pero reference_date dice que la 24 cerro
    idx = pd.date_range(end="2026-09-23", periods=10, freq="B")
    df = pd.DataFrame({("Close", "AAPL"): [100.0] * 10}, index=idx)
    ref = datetime(2026, 9, 24, 15, 59, tzinfo=ZoneInfo("UTC"))
    result = _compute_by_market(df, ref)

    # last_closed_session deberia ser 2026-09-24
    assert result["US_EQUITY"]["last_closed_session"] == "2026-09-24"
    # Como la 24 no esta en el indice -> coverage 0 -> INVALID
    assert result["US_EQUITY"]["coverage_at_session"] == 0
    assert result["US_EQUITY"]["status"] == "INVALID"


def test_bymarket_status_valid_con_cobertura_completa(monkeypatch):
    from src import instrument_registry as ir
    from src import market_hours as mh
    monkeypatch.setattr(ir, "get_market", lambda t: "US_EQUITY")
    monkeypatch.setattr(mh, "is_session_closed",
                        lambda market, sd, ref: sd.weekday() < 5)
    monkeypatch.setattr(mh, "is_trading_session", lambda m, d: d.weekday() < 5)

    df = _make_df_multi_market({"US_EQUITY": ["AAPL", "MSFT"]},
                                end_date="2026-09-24")
    ref = datetime(2026, 9, 24, 15, 59, tzinfo=ZoneInfo("UTC"))
    result = _compute_by_market(df, ref)
    assert result["US_EQUITY"]["status"] == "VALID"
    assert result["US_EQUITY"]["coverage_at_session"] == 1.0


def test_bymarket_punto5_unknown_con_n_mayor_cero_invalid(monkeypatch):
    """Punto 5 del dictamen: UNKNOWN con n>0 -> INVALID, no SKIP."""
    from src import instrument_registry as ir
    from src import market_hours as mh
    monkeypatch.setattr(ir, "get_market", lambda t: "UNKNOWN")
    monkeypatch.setattr(mh, "is_session_closed",
                        lambda market, sd, ref: sd.weekday() < 5)
    monkeypatch.setattr(mh, "is_trading_session", lambda m, d: d.weekday() < 5)

    df = _make_df_multi_market({"UNKNOWN": ["MYSTERY"]})
    ref = datetime(2026, 9, 24, 15, 59, tzinfo=ZoneInfo("UTC"))
    result = _compute_by_market(df, ref)
    assert result["UNKNOWN"]["n"] == 1
    assert result["UNKNOWN"]["status"] == "INVALID"


def test_bymarket_skip_solo_si_n_cero(monkeypatch):
    """Mercados conocidos sin tickers -> SKIP."""
    from src import instrument_registry as ir
    from src import market_hours as mh
    monkeypatch.setattr(ir, "get_market", lambda t: "US_EQUITY")
    monkeypatch.setattr(mh, "is_session_closed",
                        lambda market, sd, ref: sd.weekday() < 5)
    monkeypatch.setattr(mh, "is_trading_session", lambda m, d: d.weekday() < 5)

    df = _make_df_multi_market({"US_EQUITY": ["AAPL"]})
    ref = datetime(2026, 9, 24, 15, 59, tzinfo=ZoneInfo("UTC"))
    result = _compute_by_market(df, ref)
    # Xetra no tiene tickers en este df -> SKIP
    assert result["XETRA"]["n"] == 0
    assert result["XETRA"]["status"] == "SKIP"


def test_bymarket_punto6_universo_dinamico(monkeypatch):
    """Punto 6: el universo de mercados se hereda de _KNOWN_MARKETS."""
    from src import instrument_registry as ir
    from src import market_hours as mh
    monkeypatch.setattr(ir, "get_market", lambda t: "US_EQUITY")
    monkeypatch.setattr(mh, "is_session_closed",
                        lambda market, sd, ref: sd.weekday() < 5)
    monkeypatch.setattr(mh, "is_trading_session", lambda m, d: d.weekday() < 5)

    df = _make_df_multi_market({"US_EQUITY": ["AAPL"]})
    ref = datetime(2026, 9, 24, 15, 59, tzinfo=ZoneInfo("UTC"))
    result = _compute_by_market(df, ref)
    # Todos los mercados de _KNOWN_MARKETS deben aparecer
    for m in mh._KNOWN_MARKETS:
        assert m in result


def test_bymarket_df_sin_multiindex_devuelve_vacio():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    ref = datetime(2026, 9, 24, 15, 59, tzinfo=ZoneInfo("UTC"))
    assert _compute_by_market(df, ref) == {}


def test_bymarket_reference_date_tz_naive_devuelve_vacio():
    df = pd.DataFrame({("Close", "AAPL"): [100.0] * 5})
    ref = datetime(2026, 9, 24, 15, 59)  # sin tz
    assert _compute_by_market(df, ref) == {}


def test_bymarket_coverage_parcial(monkeypatch):
    """Un mercado con 2 de 3 tickers con Close -> coverage 0.667."""
    from src import instrument_registry as ir
    from src import market_hours as mh
    monkeypatch.setattr(ir, "get_market", lambda t: "XETRA")
    monkeypatch.setattr(mh, "is_session_closed",
                        lambda market, sd, ref: sd.weekday() < 5)
    monkeypatch.setattr(mh, "is_trading_session", lambda m, d: d.weekday() < 5)

    idx = pd.date_range(end="2026-09-24", periods=5, freq="B")
    df = pd.DataFrame({
        ("Close", "A"): [100.0] * 5,
        ("Close", "B"): [100.0] * 5,
        ("Close", "C"): [100.0, 100.0, 100.0, 100.0, np.nan],
    }, index=idx)
    ref = datetime(2026, 9, 24, 15, 59, tzinfo=ZoneInfo("UTC"))
    result = _compute_by_market(df, ref)
    assert result["XETRA"]["n"] == 3
    assert abs(result["XETRA"]["coverage_at_session"] - (2/3)) < 1e-4
    assert result["XETRA"]["status"] == "VALID_WITH_MISSING"


def test_bymarket_integrado_en_manifest(tmp_path, monkeypatch):
    """E2E: write_artifact_with_manifest incluye by_market."""
    from src import instrument_registry as ir
    from src import market_hours as mh
    from src import utils as u
    import json

    monkeypatch.setattr(ir, "get_market", lambda t: "US_EQUITY")
    monkeypatch.setattr(mh, "is_session_closed",
                        lambda market, sd, ref: sd.weekday() < 5)
    monkeypatch.setattr(mh, "is_trading_session", lambda m, d: d.weekday() < 5)

    df = _make_df_multi_market({"US_EQUITY": ["AAPL", "MSFT"]},
                                end_date="2026-09-24")
    pq = tmp_path / "test.parquet"
    ref = datetime(2026, 9, 24, 15, 59, tzinfo=ZoneInfo("UTC"))

    manifest = u.write_artifact_with_manifest(
        df, str(pq), source="test", reference_date=ref, run_id="20260924_120000"
    )
    assert manifest
    assert "by_market" in manifest["quality"]
    bm = manifest["quality"]["by_market"]
    assert "US_EQUITY" in bm
    # Verificar que se escribio a disco
    mf = Path(str(pq) + ".manifest.json")
    assert mf.exists()
    loaded = json.loads(mf.read_text(encoding="utf-8"))
    assert "by_market" in loaded["quality"]