"""Tests de target_universe (sin red)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.institutional_accumulation.identity.target_universe import (
    COLUMNS,
    build_scf_index,
    membership_summary,
    resolve_cusips,
)


def _catalog() -> pd.DataFrame:
    return pd.DataFrame([
        {"radar_ticker": "AAPL", "share_class_figi": "SC_AAPL"},
        {"radar_ticker": "ABBV", "share_class_figi": "SC_ABBV"},
        {"radar_ticker": "MISS", "share_class_figi": None},
    ])


def test_build_scf_index_ignora_nulos():
    idx = build_scf_index(_catalog())
    assert idx == {"SC_AAPL": "AAPL", "SC_ABBV": "ABBV"}


def test_resolve_cusip_target(monkeypatch):
    fake = {
        "037833100": {"ok": True, "data": [
            {"figi": "F", "shareClassFIGI": "SC_AAPL", "ticker": "AAPL",
             "securityType": "Common Stock", "marketSector": "Equity", "exchCode": "US"}
        ], "error": None},
    }
    monkeypatch.setattr(
        "src.institutional_accumulation.identity.target_universe.map_identifiers",
        lambda *a, **k: fake,
    )
    df = resolve_cusips(["037833100"], _catalog(), source_date="2026-09-19")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["radar_ticker"] == "AAPL"
    assert row["target_membership"] == True
    assert row["status"] == "OK"
    assert list(df.columns) == list(COLUMNS)


def test_resolve_cusip_no_en_radar(monkeypatch):
    fake = {
        "999999999": {"ok": True, "data": [
            {"figi": "F", "shareClassFIGI": "SC_OTHER", "ticker": "OTHER",
             "securityType": "Common Stock", "marketSector": "Equity", "exchCode": "US"}
        ], "error": None},
    }
    monkeypatch.setattr(
        "src.institutional_accumulation.identity.target_universe.map_identifiers",
        lambda *a, **k: fake,
    )
    df = resolve_cusips(["999999999"], _catalog(), source_date="2026-09-19")
    row = df.iloc[0]
    assert row["target_membership"] == False
    assert row["radar_ticker"] is None
    assert row["status"] == "NOT_IN_RADAR"


def test_resolve_cusip_no_id(monkeypatch):
    fake = {"111111111": {"ok": False, "data": None, "error": "No identifier found."}}
    monkeypatch.setattr(
        "src.institutional_accumulation.identity.target_universe.map_identifiers",
        lambda *a, **k: fake,
    )
    df = resolve_cusips(["111111111"], _catalog(), source_date="2026-09-19")
    assert df.iloc[0]["status"] == "NO_ID"
    assert df.iloc[0]["target_membership"] == False


def test_resolve_cusip_error(monkeypatch):
    fake = {"222222222": {"ok": False, "data": None, "error": "HTTP 500"}}
    monkeypatch.setattr(
        "src.institutional_accumulation.identity.target_universe.map_identifiers",
        lambda *a, **k: fake,
    )
    df = resolve_cusips(["222222222"], _catalog(), source_date="2026-09-19")
    assert df.iloc[0]["status"] == "ERROR"


def test_resolve_ordena_y_dedup(monkeypatch):
    fake = {}
    monkeypatch.setattr(
        "src.institutional_accumulation.identity.target_universe.map_identifiers",
        lambda *a, **k: fake,
    )
    df = resolve_cusips(["B", "A", "A", "B"], _catalog(), source_date="2026-09-19")
    assert list(df["cusip"]) == ["A", "B"]


def test_membership_summary(monkeypatch):
    fake = {
        "A": {"ok": True, "data": [
            {"shareClassFIGI": "SC_AAPL", "exchCode": "US"}
        ], "error": None},
        "B": {"ok": True, "data": [
            {"shareClassFIGI": "SC_OTHER", "exchCode": "US"}
        ], "error": None},
        "C": {"ok": False, "data": None, "error": "No identifier found."},
    }
    monkeypatch.setattr(
        "src.institutional_accumulation.identity.target_universe.map_identifiers",
        lambda *a, **k: fake,
    )
    df = resolve_cusips(["A", "B", "C"], _catalog(), source_date="2026-09-19")
    s = membership_summary(df)
    assert s["n_total"] == 3
    assert s["n_target"] == 1
    assert s["n_not_in_radar"] == 1
    assert s["n_no_id"] == 1
    assert s["pct_target"] == round(100.0 * 1 / 3, 4)


def test_membership_summary_vacio():
    s = membership_summary(pd.DataFrame(columns=list(COLUMNS)))
    assert s["n_total"] == 0
    assert s["pct_target"] == 0.0
