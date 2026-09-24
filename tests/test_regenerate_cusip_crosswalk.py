"""Tests de scripts/regenerate_cusip_crosswalk.py (sin red)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import regenerate_cusip_crosswalk as rcc


# --- helpers ---

def _mk_infotable(tmp_path: Path, quarter: str, rows: list) -> Path:
    """Crea processed/<q>/INFOTABLE.parquet con CUSIP/FIGI/TITLEOFCLASS."""
    d = tmp_path / quarter
    d.mkdir(parents=True, exist_ok=True)
    p = d / "INFOTABLE.parquet"
    pd.DataFrame(rows, columns=["CUSIP", "FIGI", "TITLEOFCLASS"]).to_parquet(p)
    return p


def _mk_catalog(path: Path, mapping: dict) -> None:
    """Escribe radar_target_catalog.csv con {scf: ticker}."""
    rows = [{"radar_ticker": tk, "share_class_figi": scf}
            for scf, tk in mapping.items()]
    pd.DataFrame(rows).to_csv(path, index=False)

# --- _period_end ---

@pytest.mark.parametrize("q,expected", [
    ("2025Q4", "2025-12-31"),
    ("2026Q1", "2026-03-31"),
    ("2026Q2", "2026-06-30"),
    ("2026Q3", "2026-09-30"),
])
def test_period_end(q, expected):
    assert rcc._period_end(q) == expected


# --- _discover_quarters ---

def test_discover_quarters(tmp_path):
    _mk_infotable(tmp_path, "2025Q4", [])
    _mk_infotable(tmp_path, "2026Q1", [])
    (tmp_path / "2026Q2").mkdir()
    (tmp_path / "2026Q2" / "INFOTABLE.parquet").write_bytes(b"")
    (tmp_path / "garbage").mkdir()
    out = rcc._discover_quarters(tmp_path)
    assert out == ["2025Q4", "2026Q1", "2026Q2"]


def test_discover_quarters_vacio(tmp_path):
    assert rcc._discover_quarters(tmp_path) == []
    assert rcc._discover_quarters(tmp_path / "nope") == []


# --- _load_radar_index ---

def test_load_radar_index(tmp_path):
    cat = tmp_path / "catalog.csv"
    _mk_catalog(cat, {"SC_AAPL": "AAPL", "SC_MSFT": "MSFT"})
    idx = rcc._load_radar_index(cat)
    assert idx == {"SC_AAPL": "AAPL", "SC_MSFT": "MSFT"}


# --- _extract_observations ---

def test_extract_filtra_por_radar(tmp_path):
    p = _mk_infotable(tmp_path, "2026Q1", [
        ("037833100", "SC_AAPL", "COM"),
        ("999999999", "SC_OTHER", "COM"),  # no en radar
        ("00287Y109", "SC_ABBV", "COM"),
    ])
    idx = {"SC_AAPL": "AAPL", "SC_ABBV": "ABBV"}
    obs = rcc._extract_observations("2026Q1", p, idx)
    assert set(obs.keys()) == {("037833100", "AAPL"), ("00287Y109", "ABBV")}
    assert obs[("037833100", "AAPL")]["title"] == "COM"


def test_extract_ignora_figi_nan(tmp_path):
    p = _mk_infotable(tmp_path, "2026Q1", [
        ("037833100", None, "COM"),
        ("00287Y109", "SC_ABBV", "COM"),
    ])
    idx = {"SC_AAPL": "AAPL", "SC_ABBV": "ABBV"}
    obs = rcc._extract_observations("2026Q1", p, idx)
    assert len(obs) == 1


# --- _build_auto_rows ---

def test_build_auto_rows_valid_to_null_si_ultimo():
    obs = {("037833100", "AAPL"): {"title": "COM", "quarters": ["2026Q2"]}}
    rows = rcc._build_auto_rows(obs, latest_quarter="2026Q2")
    assert rows[0]["valid_from"] == "2026-06-30"
    assert rows[0]["valid_to"] == ""
    assert rows[0]["verified_by"] == "auto"


def test_build_auto_rows_valid_to_si_no_ultimo():
    obs = {("037833100", "AAPL"): {"title": "COM", "quarters": ["2025Q4"]}}
    rows = rcc._build_auto_rows(obs, latest_quarter="2026Q2")
    assert rows[0]["valid_from"] == "2025-12-31"
    assert rows[0]["valid_to"] == "2025-12-31"


# --- _merge ---

def test_merge_manual_gana():
    manual = pd.DataFrame([{
        "CUSIP": "037833100", "ticker": "AAPL_MANUAL", "valid_from": "2020-01-01",
        "valid_to": "", "source": "SEC-EDGAR", "reason": "manual",
        "title_of_class": "COM", "verified_by": "manual",
    }])
    auto = [{
        "CUSIP": "037833100", "ticker": "AAPL", "valid_from": "2025-12-31",
        "valid_to": "", "source": "SEC-EDGAR", "reason": "auto",
        "title_of_class": "COM", "verified_by": "auto",
    }]
    out = rcc._merge(manual, auto)
    assert len(out) == 1
    assert out.iloc[0]["ticker"] == "AAPL_MANUAL"


# --- _validate ---

def test_validate_ok():
    df = pd.DataFrame([{
        "CUSIP": "037833100", "ticker": "AAPL", "valid_from": "2025-12-31",
        "valid_to": "2026-03-31", "source": "SEC-EDGAR", "reason": "x",
        "title_of_class": "COM", "verified_by": "auto",
    }])
    rcc._validate(df)  # no lanza


def test_validate_duplicado_falla():
    df = pd.DataFrame([
        {"CUSIP": "037833100", "ticker": "AAPL", "valid_from": "2025-12-31",
         "valid_to": "", "source": "SEC-EDGAR", "reason": "x",
         "title_of_class": "COM", "verified_by": "auto"},
        {"CUSIP": "037833100", "ticker": "AAPL2", "valid_from": "2025-12-31",
         "valid_to": "", "source": "SEC-EDGAR", "reason": "x",
         "title_of_class": "COM", "verified_by": "auto"},
    ])
    with pytest.raises(ValueError):
        rcc._validate(df)


def test_validate_valid_from_mayor_falla():
    df = pd.DataFrame([{
        "CUSIP": "037833100", "ticker": "AAPL", "valid_from": "2026-06-30",
        "valid_to": "2025-12-31", "source": "SEC-EDGAR", "reason": "x",
        "title_of_class": "COM", "verified_by": "auto",
    }])
    with pytest.raises(ValueError):
        rcc._validate(df)


# --- main sin trimestres ---

def test_main_aborta_sin_trimestres(tmp_path, monkeypatch):
    monkeypatch.setattr(rcc, "DATA_DIR", tmp_path / "nope")
    monkeypatch.setattr("sys.argv", ["regenerate_cusip_crosswalk.py"])
    assert rcc.main() == 1