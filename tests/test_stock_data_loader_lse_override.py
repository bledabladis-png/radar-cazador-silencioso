# -*- coding: utf-8 -*-
"""Tests de _apply_lse_close_override (subciclo 2b).

Replican los casos del test quirurgico con filesystem aislado via
monkeypatch sobre src.stock_data_loader.
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.stock_data_loader as sdl
from src.stock_data_loader import _apply_lse_close_override


MAD = ZoneInfo("Europe/Madrid")
SESSION = "2026-09-24"
REF = datetime(2026, 9, 25, 12, 0, tzinfo=MAD)


def _make_data(tickers, session=SESSION):
    idx = pd.DatetimeIndex([pd.Timestamp(session)])
    cols = pd.MultiIndex.from_product(
        [["Close", "Open", "High", "Low", "Volume"], tickers]
    )
    row = []
    for field in ["Close", "Open", "High", "Low", "Volume"]:
        for _ in tickers:
            if field == "Close":
                row.append(float("nan"))
            elif field == "Volume":
                row.append(1_000_000)
            else:
                row.append(99.0)
    return pd.DataFrame([row], index=idx, columns=cols)


def _write_scraper_json(datos_dir, ric, session, close, open_=99.0,
                         high=99.0, low=99.0):
    datos_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "request": {"ric": ric},
        "data": [{
            "_DATE_END": session,
            "OPEN_PRC": str(open_),
            "HIGH_1": str(high),
            "LOW_1": str(low),
            "CLOSE_PRC": str(close),
        }],
        "status": "OK",
    }
    (datos_dir / (ric.replace(".", "_") + ".json")).write_text(
        json.dumps(payload), encoding="utf-8"
    )


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Aisla DATOS_DIR y PROVENANCE_PATH en tmp_path."""
    datos_dir = tmp_path / "lse_close" / "datos"
    prov_path = tmp_path / "lse_close_provenance.json"
    monkeypatch.setattr(sdl, "LSE_SCRAPER_DATOS_DIR", str(datos_dir))
    monkeypatch.setattr(sdl, "LSE_SCRAPER_PROVENANCE_PATH", str(prov_path))
    return {"datos_dir": datos_dir, "prov_path": prov_path}


# ---------------- Caso 1: scraper ausente ----------------

def test_scraper_ausente_no_override(isolated):
    data = _make_data(["AZN.L", "HSBA.L"])
    data, stats = _apply_lse_close_override(
        data, REF, "run1", SESSION
    )
    assert stats["provenance_status"] == "UNAVAILABLE"
    assert stats["applied"] == []
    assert pd.isna(data.loc[SESSION, ("Close", "AZN.L")])


def test_scraper_ausente_provenance_escrita(isolated):
    data = _make_data(["AZN.L"])
    _apply_lse_close_override(data, REF, "run1", SESSION)
    assert isolated["prov_path"].exists()
    p = json.loads(isolated["prov_path"].read_text(encoding="utf-8"))
    assert p["status"] == "UNAVAILABLE"
    assert p["scraper_available"] is False


def test_scraper_ausente_tickers_from_yahoo(isolated):
    data = _make_data(["AZN.L", "HSBA.L"])
    _, stats = _apply_lse_close_override(data, REF, "run1", SESSION)
    assert stats["tickers_from_yahoo"] == ["AZN.L", "HSBA.L"]


# ---------------- Caso 2: scraper completo ----------------

def test_scraper_completo_override_aplica(isolated):
    _write_scraper_json(isolated["datos_dir"], "AZN.L", SESSION, 12400.0)
    _write_scraper_json(isolated["datos_dir"], "HSBA.L", SESSION, 1499.8)

    data = _make_data(["AZN.L", "HSBA.L"])
    data, stats = _apply_lse_close_override(data, REF, "run2", SESSION)

    assert stats["provenance_status"] == "OK"
    assert stats["applied"] == ["AZN.L", "HSBA.L"]
    assert data.loc[SESSION, ("Close", "AZN.L")] == 12400.0
    assert data.loc[SESSION, ("Close", "HSBA.L")] == 1499.8


def test_scraper_completo_no_toca_ohlcv(isolated):
    _write_scraper_json(isolated["datos_dir"], "AZN.L", SESSION, 12400.0)
    data = _make_data(["AZN.L"])
    data, _ = _apply_lse_close_override(data, REF, "run2", SESSION)
    assert data.loc[SESSION, ("Open", "AZN.L")] == 99.0
    assert data.loc[SESSION, ("High", "AZN.L")] == 99.0
    assert data.loc[SESSION, ("Low", "AZN.L")] == 99.0
    assert data.loc[SESSION, ("Volume", "AZN.L")] == 1_000_000


def test_scraper_completo_provenance_ok(isolated, monkeypatch):
    monkeypatch.setenv("LSE_SCRAPER_COMMIT", "sha_abc123")
    _write_scraper_json(isolated["datos_dir"], "AZN.L", SESSION, 12400.0)
    data = _make_data(["AZN.L"])
    _apply_lse_close_override(data, REF, "run2", SESSION)
    p = json.loads(isolated["prov_path"].read_text(encoding="utf-8"))
    assert p["status"] == "OK"
    assert p["source_commit"] == "sha_abc123"
    assert p["tickers_from_scraper"] == ["AZN.L"]
    assert p["scraper_available"] is True
    assert p["scraper_used"] is True


# ---------------- Caso 3: scraper parcial ----------------

def test_scraper_parcial_status_partial(isolated, monkeypatch):
    monkeypatch.setenv("LSE_SCRAPER_COMMIT", "sha_abc123")
    _write_scraper_json(isolated["datos_dir"], "AZN.L", SESSION, 12400.0)
    # HSBA.L no tiene JSON -> from_yahoo -> missing

    data = _make_data(["AZN.L", "HSBA.L"])
    data, stats = _apply_lse_close_override(data, REF, "run3", SESSION)

    assert stats["provenance_status"] == "PARTIAL"
    assert stats["applied"] == ["AZN.L"]
    assert stats["tickers_from_yahoo"] == ["HSBA.L"]
    assert stats["tickers_missing"] == ["HSBA.L"]


def test_scraper_sin_observacion_para_sesion(isolated):
    """JSON existe pero sin la fila de la sesion esperada -> NO_COVERAGE."""
    _write_scraper_json(isolated["datos_dir"], "AZN.L", "2026-09-23", 12300.0)
    data = _make_data(["AZN.L"])
    _, stats = _apply_lse_close_override(data, REF, "run4", SESSION)
    assert stats["provenance_status"] == "NO_COVERAGE"
    assert stats["applied"] == []
    assert stats["tickers_missing"] == ["AZN.L"]


# ---------------- Casos borde ----------------

def test_data_vacio_no_op(isolated):
    data = pd.DataFrame()
    data_out, stats = _apply_lse_close_override(data, REF, "run5", SESSION)
    assert data_out is data
    assert stats["applied"] == []


def test_sin_tickers_lse_no_op(isolated):
    data = _make_data(["AAPL", "MSFT"])
    _, stats = _apply_lse_close_override(data, REF, "run6", SESSION)
    assert stats["applied"] == []


def test_provenance_falla_no_bloquea(isolated, monkeypatch):
    """Si write_lse_provenance lanza ValueError, no debe romper el pipeline."""
    _write_scraper_json(isolated["datos_dir"], "AZN.L", SESSION, 12400.0)
    # Sin LSE_SCRAPER_COMMIT -> scraper_used=True + commit vacio -> ValueError
    monkeypatch.delenv("LSE_SCRAPER_COMMIT", raising=False)

    data = _make_data(["AZN.L"])
    # No debe lanzar; solo avisar por consola
    data, stats = _apply_lse_close_override(data, REF, "run7", SESSION)
    assert stats["applied"] == ["AZN.L"]
    assert data.loc[SESSION, ("Close", "AZN.L")] == 12400.0
