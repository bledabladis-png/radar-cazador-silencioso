"""Tests del RADAR_TARGET_CATALOG (sin red)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.institutional_accumulation.identity.radar_target_catalog import (
    COLUMNS,
    SOURCE_NAME,
    build_from_probe_result,
    coverage_summary,
    load_radar_tickers,
    write_catalog,
)


def _sample_json(tmp_path: Path) -> Path:
    data = {
        "AAPL": {
            "ok": True,
            "data": [{
                "figi": "BBG000B9XRY4",
                "shareClassFIGI": "BBG001S5N8V8",
                "compositeFIGI": "BBG000B9XRY4",
                "ticker": "AAPL",
                "name": "APPLE INC",
                "securityType": "Common Stock",
                "marketSector": "Equity",
                "exchCode": "US",
            }],
            "error": None,
        },
        "XYZ": {
            "ok": True,
            "data": [{
                "figi": "BBG000XYZ123",
                "ticker": "XYZ",
                "name": "XYZ CORP",
                "exchCode": "US",
            }],
            "error": None,
        },
        "MISS": {"ok": False, "data": None, "error": "No identifier found."},
    }
    p = tmp_path / "sample.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_build_from_probe_result_columnas(tmp_path):
    df = build_from_probe_result(_sample_json(tmp_path), source_date="2026-09-19")
    assert list(df.columns) == list(COLUMNS)


def test_build_status_ok_partial_miss(tmp_path):
    df = build_from_probe_result(_sample_json(tmp_path), source_date="2026-09-19")
    by = df.set_index("radar_ticker")["status"].to_dict()
    assert by["AAPL"] == "OK"
    assert by["XYZ"] == "PARTIAL"
    assert by["MISS"] == "MISS"


def test_build_source_poblado(tmp_path):
    df = build_from_probe_result(_sample_json(tmp_path), source_date="2026-09-19")
    assert (df["source"] == SOURCE_NAME).all()
    assert (df["source_date"] == "2026-09-19").all()


def test_build_ordenado_por_ticker(tmp_path):
    df = build_from_probe_result(_sample_json(tmp_path), source_date="2026-09-19")
    assert list(df["radar_ticker"]) == sorted(df["radar_ticker"])


def test_coverage_summary(tmp_path):
    df = build_from_probe_result(_sample_json(tmp_path), source_date="2026-09-19")
    s = coverage_summary(df)
    assert s["n_total"] == 3
    assert s["n_ok"] == 1
    assert s["n_partial"] == 1
    assert s["n_miss"] == 1
    assert s["pct_with_scf"] == round(100.0 * 1 / 3, 4)


def test_coverage_summary_vacio():
    s = coverage_summary(pd.DataFrame(columns=list(COLUMNS)))
    assert s["n_total"] == 0
    assert s["pct_with_scf"] == 0.0


def test_write_catalog_round_trip(tmp_path):
    df = build_from_probe_result(_sample_json(tmp_path), source_date="2026-09-19")
    out = tmp_path / "catalog.csv"
    write_catalog(df, out)
    back = pd.read_csv(out, dtype=str)
    assert len(back) == len(df)
    assert list(back.columns) == list(COLUMNS)
    # sin BOM: primer byte no es EF BB BF
    first = out.read_bytes()[:3]
    assert first != b"\xef\xbb\xbf"


def test_write_catalog_crea_directorio(tmp_path):
    df = build_from_probe_result(_sample_json(tmp_path), source_date="2026-09-19")
    out = tmp_path / "sub" / "deep" / "catalog.csv"
    write_catalog(df, out)
    assert out.exists()
