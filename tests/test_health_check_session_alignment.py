# -*- coding: utf-8 -*-
"""Test N-C7: alineacion de sesion esperada en health_check.

Dictamen auditor externo 2026-10-16: anadido control explicito que
compara last_date vs expected_session del manifest, y detecta mercados
INVALID en su sesion esperada.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.health_check as hc


def _mk_manifest(tmp_path, name, expected, last, status, by_market=None):
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    q = {"expected_session": expected, "last_date": last, "status": status}
    if by_market:
        q["by_market"] = by_market
    p = d / f"{name}.parquet.manifest.json"
    p.write_text(json.dumps({"quality": q}), encoding="utf-8")


def _run(tmp_path):
    with patch.object(hc, "PROJECT_ROOT", tmp_path):
        return hc.check_expected_session_alignment()


def test_aligned_ok(tmp_path):
    _mk_manifest(tmp_path, "stock_prices", "2026-10-15", "2026-10-15", "VALID")
    _mk_manifest(tmp_path, "market_data", "2026-10-15", "2026-10-15", "VALID")
    res = _run(tmp_path)
    assert all(r.status == hc.OK for r in res)


def test_lagged_sin_invalid_warn(tmp_path):
    _mk_manifest(tmp_path, "stock_prices", "2026-10-15", "2026-10-15", "VALID")
    _mk_manifest(tmp_path, "market_data", "2026-10-15", "2026-10-14",
                 "VALID_WITH_MISSING")
    res = _run(tmp_path)
    mr = [r for r in res if "market_data" in r.name][0]
    assert mr.status == hc.WARN
    assert "LAGGED" in mr.message


def test_lagged_con_invalid_fail(tmp_path):
    _mk_manifest(tmp_path, "stock_prices", "2026-10-15", "2026-10-15", "VALID")
    _mk_manifest(tmp_path, "market_data", "2026-10-15", "2026-10-14",
                 "VALID_WITH_MISSING",
                 by_market={"US_EQUITY": {"n": 563, "status": "INVALID",
                                          "coverage_at_session": 0.0}})
    res = _run(tmp_path)
    mr = [r for r in res if "market_data" in r.name][0]
    assert mr.status == hc.FAIL
    assert "US_EQUITY" in mr.message


def test_manifest_ausente(tmp_path):
    res = _run(tmp_path)
    assert all(r.status == hc.SKIP for r in res)


def test_aligned_pero_invalid_fail(tmp_path):
    _mk_manifest(tmp_path, "stock_prices", "2026-10-15", "2026-10-15", "INVALID")
    _mk_manifest(tmp_path, "market_data", "2026-10-15", "2026-10-15", "VALID")
    res = _run(tmp_path)
    sr = [r for r in res if "stock_prices" in r.name][0]
    assert sr.status == hc.FAIL