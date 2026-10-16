# -*- coding: utf-8 -*-
"""Test M3: alineacion con sesion esperada desde manifests.

Dictamen auditor externo 2026-10-16: separar "frescura por antiguedad"
de "alineacion con sesion esperada". La alineacion se deriva de la
misma autoridad que usa el Guard (manifest).
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.report.freshness as fr


def _mk_manifest(tmp_path, name, expected, last):
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    p = d / f"{name}.parquet.manifest.json"
    p.write_text(json.dumps({
        "quality": {"expected_session": expected, "last_date": last}
    }), encoding="utf-8")
    return p


def test_aligned(tmp_path, monkeypatch):
    _mk_manifest(tmp_path, "stock_prices", "2026-10-15", "2026-10-15")
    _mk_manifest(tmp_path, "market_data", "2026-10-15", "2026-10-15")
    monkeypatch.chdir(tmp_path)
    out = "".join(fr._render_session_alignment())
    assert out.count("ALIGNED") == 2


def test_lagged(tmp_path, monkeypatch):
    _mk_manifest(tmp_path, "stock_prices", "2026-10-15", "2026-10-15")
    _mk_manifest(tmp_path, "market_data", "2026-10-15", "2026-10-14")
    monkeypatch.chdir(tmp_path)
    out = "".join(fr._render_session_alignment())
    assert "LAGGED" in out
    assert "observada=2026-10-14" in out
    assert "ALIGNED" in out


def test_manifest_ausente(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = "".join(fr._render_session_alignment())
    assert "manifest ausente" in out


def test_anomaly(tmp_path, monkeypatch):
    _mk_manifest(tmp_path, "stock_prices", "2026-10-14", "2026-10-15")
    _mk_manifest(tmp_path, "market_data", "2026-10-14", "2026-10-14")
    monkeypatch.chdir(tmp_path)
    out = "".join(fr._render_session_alignment())
    assert "ANOMALY" in out