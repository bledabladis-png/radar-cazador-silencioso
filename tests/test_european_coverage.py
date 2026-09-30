# -*- coding: utf-8 -*-
"""D18 (2026-09-30): tests de src/european_coverage.

Complementa test_european_coverage_atomic.py (que cubre el fallo de
to_csv en _append_csv). Aqui se cubren las funciones puras y el
flujo normal.

Cobertura objetivo:
- _ref_to_date: normalizacion de tipos (None, date, datetime, Timestamp).
- _collect: 3 estados (OK, REVISAR, SIN_DATOS) con providers mockeados.
- _render_markdown: cabecera, conteos, casos vacios.
- _append_csv: append a fichero nuevo, dedup por date+ticker.
- generate_european_coverage_report: integracion end-to-end local.
"""
import datetime as dt
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import european_coverage as ec


# ---------- _ref_to_date ----------
def test_ref_to_date_none_devuelve_hoy():
    d = ec._ref_to_date(None)
    assert isinstance(d, dt.date)


def test_ref_to_date_date_passthrough():
    d = dt.date(2026, 9, 30)
    assert ec._ref_to_date(d) == d


def test_ref_to_date_datetime_a_date():
    d = dt.datetime(2026, 9, 30, 12, 0, 0)
    assert ec._ref_to_date(d) == dt.date(2026, 9, 30)


def test_ref_to_date_timestamp_a_date():
    ts = pd.Timestamp("2026-09-30")
    assert ec._ref_to_date(ts) == dt.date(2026, 9, 30)


def test_ref_to_date_tz_aware():
    ts = pd.Timestamp("2026-09-30 08:00:00+02:00")
    assert ec._ref_to_date(ts) == dt.date(2026, 9, 30)


# ---------- _render_markdown ----------
def test_render_markdown_cabecera_y_conteos():
    rows = [
        {"ticker": "AIR.PA", "source": "Euronext",
         "last_date": pd.Timestamp("2026-09-29"), "days_gap": 1, "status": "OK"},
        {"ticker": "SAN.MC", "source": "BME",
         "last_date": pd.Timestamp("2026-09-01"), "days_gap": 29, "status": "REVISAR"},
        {"ticker": "SAP.DE", "source": "Xetra",
         "last_date": None, "days_gap": None, "status": "SIN_DATOS"},
    ]
    md = ec._render_markdown(rows, "2026-09-30")
    assert "# Cobertura Europea - 2026-09-30" in md
    assert "Tickers europeos monitoreados: 3" in md
    assert "Cubiertos (gap <= 7 dias): 1" in md
    assert "Con huecos > 7 dias: 1" in md
    assert "Sin datos en cache: 1" in md
    assert "AIR.PA" in md
    assert "SAN.MC" in md
    assert "SAP.DE" in md


def test_render_markdown_sin_revisar_ni_sin_datos():
    rows = [
        {"ticker": "AIR.PA", "source": "Euronext",
         "last_date": pd.Timestamp("2026-09-29"), "days_gap": 1, "status": "OK"},
    ]
    md = ec._render_markdown(rows, "2026-09-30")
    assert "(vacio - sin alertas)" in md
    assert "(vacio - todos tienen cache)" in md


# ---------- _append_csv ----------
def _make_row(ticker="AIR.PA", source="Euronext", status="OK"):
    return {"ticker": ticker, "source": source,
            "last_date": pd.Timestamp("2026-09-29"),
            "days_gap": 1, "status": status}


def test_append_csv_crea_fichero_nuevo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = Path("outputs/history/european_coverage.csv")
    monkeypatch.setattr(ec, "OUTPUT_CSV", target)
    ec._append_csv([_make_row()], "2026-09-30")
    assert target.exists()
    df = pd.read_csv(target)
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AIR.PA"
    assert df.iloc[0]["date"] == "2026-09-30"


def test_append_csv_dedup_mismo_date_ticker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = Path("outputs/history/european_coverage.csv")
    monkeypatch.setattr(ec, "OUTPUT_CSV", target)
    ec._append_csv([_make_row(status="REVISAR")], "2026-09-30")
    ec._append_csv([_make_row(status="OK")], "2026-09-30")
    df = pd.read_csv(target)
    # dedup por (date, ticker) con keep='last' -> status=OK
    assert len(df) == 1
    assert df.iloc[0]["status"] == "OK"


def test_append_csv_vacio_no_hace_nada(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = Path("outputs/history/european_coverage.csv")
    monkeypatch.setattr(ec, "OUTPUT_CSV", target)
    ec._append_csv([], "2026-09-30")
    assert not target.exists()


# ---------- _collect ----------
class _FakeProvider:
    def __init__(self, tickers, data_map):
        self._tickers = tickers
        self._data = data_map

    def supported_tickers(self):
        return self._tickers

    def _load_cache(self, ticker):
        return self._data.get(ticker)


def _mock_providers():
    """Sustituye los 3 providers por fakes con datos controlados."""
    # Euronext: 1 OK (gap 1), 1 REVISAR (gap 30)
    # Xetra: 1 SIN_DATOS
    # BME: vacio (0 tickers)
    fake_euronext = _FakeProvider(
        ["AIR.PA", "SAN.PA"],
        {
            "AIR.PA": pd.DataFrame({"date": pd.to_datetime(["2026-09-29"])}),
            "SAN.PA": pd.DataFrame({"date": pd.to_datetime(["2026-08-31"])}),
        },
    )
    fake_xetra = _FakeProvider(["SAP.DE"], {"SAP.DE": None})
    fake_bme = _FakeProvider([], {})
    return fake_euronext, fake_xetra, fake_bme


def test_collect_3_estados(monkeypatch):
    fake_eu, fake_xe, fake_bme = _mock_providers()
    with patch.multiple(
        ec,
        EuronextProvider=lambda: fake_eu,
        XetraProvider=lambda: fake_xe,
        BMEProvider=lambda: fake_bme,
    ):
        rows = ec._collect(reference_date=dt.date(2026, 9, 30))
    assert len(rows) == 3
    by_ticker = {r["ticker"]: r for r in rows}
    assert by_ticker["AIR.PA"]["status"] == "OK"
    assert by_ticker["AIR.PA"]["days_gap"] == 1
    assert by_ticker["SAN.PA"]["status"] == "REVISAR"
    assert by_ticker["SAN.PA"]["days_gap"] == 30
    assert by_ticker["SAP.DE"]["status"] == "SIN_DATOS"
    assert by_ticker["SAP.DE"]["days_gap"] is None


# ---------- generate_european_coverage_report ----------
def test_generate_report_integracion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake_eu, fake_xe, fake_bme = _mock_providers()
    md_target = Path("outputs/audit/european_coverage.md")
    csv_target = Path("outputs/history/european_coverage.csv")
    monkeypatch.setattr(ec, "OUTPUT_MD", md_target)
    monkeypatch.setattr(ec, "OUTPUT_CSV", csv_target)

    with patch.multiple(
        ec,
        EuronextProvider=lambda: fake_eu,
        XetraProvider=lambda: fake_xe,
        BMEProvider=lambda: fake_bme,
    ):
        result = ec.generate_european_coverage_report(
            reference_date=dt.date(2026, 9, 30))

    assert result == {"total": 3, "ok": 1, "revisar": 1, "sin_datos": 1}
    assert md_target.exists()
    assert csv_target.exists()
    md = md_target.read_text(encoding="utf-8")
    assert "# Cobertura Europea - 2026-09-30" in md
