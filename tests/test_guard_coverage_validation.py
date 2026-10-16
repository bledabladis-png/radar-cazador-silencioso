# -*- coding: utf-8 -*-
"""Tests de integracion: validacion contractual en el guard (enmienda v1.1)."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.guard_coverage as gc


def _mk_manifest(tmp_path, name, by_market, status="VALID_WITH_MISSING",
                 last="2026-10-15", expected="2026-10-15",
                 cov=0.99, sha="abc"*20):
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    p = d / f"{name}.parquet.manifest.json"
    p.write_text(json.dumps({
        "artifact": {"sha256": sha},
        "quality": {
            "status": status,
            "last_date": last,
            "expected_session": expected,
            "coverage_pct_last": cov,
            "by_market": by_market,
        },
    }), encoding="utf-8")
    return p


def _fake_parquet(tmp_path, name):
    """Crea un parquet vacio y devuelve su sha256 real."""
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    p = d / f"{name}.parquet"
    p.write_bytes(b"")
    return hashlib.sha256(b"").hexdigest()


def _run_guard(monkeypatch, manifests=("data/stock_prices.parquet.manifest.json",),
               threshold=0.95):
    """Ejecuta gc.main() con sys.argv saneado."""
    monkeypatch.setattr(sys, "argv", ["guard_coverage.py"])
    monkeypatch.setattr(gc, "GUARDED_MANIFESTS", manifests)
    return gc.main()


def test_equity_eod_invalid_bloquea(tmp_path, monkeypatch):
    """Enmienda v1.1: US_EQUITY INVALID con coverage 0 -> BLOCKED."""
    sha = _fake_parquet(tmp_path, "stock_prices")
    _mk_manifest(tmp_path, "stock_prices",
                 {"US_EQUITY": {"n": 255, "status": "INVALID",
                                 "coverage_at_session": 0.0}},
                 sha=sha)
    monkeypatch.chdir(tmp_path)
    rc = _run_guard(monkeypatch)
    assert rc == 1


def test_equity_eod_ok_pasa(tmp_path, monkeypatch):
    sha = _fake_parquet(tmp_path, "stock_prices")
    _mk_manifest(tmp_path, "stock_prices",
                 {"US_EQUITY": {"n": 255, "status": "VALID",
                                 "coverage_at_session": 0.99}},
                 sha=sha)
    monkeypatch.chdir(tmp_path)
    rc = _run_guard(monkeypatch)
    assert rc == 0


def test_equity_eod_coverage_baja_bloquea(tmp_path, monkeypatch):
    """Regla del auditor: cobertura insuficiente bloquea aun con status
    distinto de INVALID."""
    sha = _fake_parquet(tmp_path, "stock_prices")
    _mk_manifest(tmp_path, "stock_prices",
                 {"US_EQUITY": {"n": 255, "status": "VALID_WITH_MISSING",
                                 "coverage_at_session": 0.50}},
                 sha=sha)
    monkeypatch.chdir(tmp_path)
    rc = _run_guard(monkeypatch)
    assert rc == 1


def test_by_market_ausente_no_bloquea(tmp_path, monkeypatch):
    """Enmienda v1.1: sin by_market -> WARN -> exit 0.
    La validacion global (cobertura, status, sha) sigue aplicando."""
    sha = _fake_parquet(tmp_path, "stock_prices")
    p = tmp_path / "data" / "stock_prices.parquet.manifest.json"
    p.write_text(json.dumps({
        "artifact": {"sha256": sha},
        "quality": {
            "status": "VALID",
            "last_date": "2026-10-15",
            "expected_session": "2026-10-15",
            "coverage_pct_last": 0.99,
        },
    }), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rc = _run_guard(monkeypatch)
    assert rc == 0