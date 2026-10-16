# -*- coding: utf-8 -*-
"""Tests de integracion: validacion contractual en el guard (enmienda v1.1)."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

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
    """Crea un parquet vacio que coincida con el sha declarado."""
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    p = d / f"{name}.parquet"
    p.write_bytes(b"")
    import hashlib
    return hashlib.sha256(b"").hexdigest()


def test_equity_eod_invalid_bloquea(tmp_path, monkeypatch):
    """Enmienda v1.1: US_EQUITY INVALID con coverage 0 -> BLOCKED."""
    sha = _fake_parquet(tmp_path, "stock_prices")
    _fake_parquet(tmp_path, "market_data")
    _mk_manifest(tmp_path, "stock_prices",
                 {"US_EQUITY": {"n": 255, "status": "INVALID",
                                 "coverage_at_session": 0.0}},
                 sha=sha)
    _mk_manifest(tmp_path, "market_data",
                 {"US_EQUITY": {"n": 255, "status": "VALID",
                                 "coverage_at_session": 0.99}},
                 sha=sha)
    monkeypatch.chdir(tmp_path)
    # Forzar el path relativo
    monkeypatch.setattr(gc, "GUARDED_MANIFESTS", (
        "data/stock_prices.parquet.manifest.json",
    ))
    rc = gc.main()
    assert rc == 1


def test_equity_eod_ok_pasa(tmp_path, monkeypatch):
    sha = _fake_parquet(tmp_path, "stock_prices")
    _mk_manifest(tmp_path, "stock_prices",
                 {"US_EQUITY": {"n": 255, "status": "VALID",
                                 "coverage_at_session": 0.99}},
                 sha=sha)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gc, "GUARDED_MANIFESTS", (
        "data/stock_prices.parquet.manifest.json",
    ))
    rc = gc.main()
    assert rc == 0