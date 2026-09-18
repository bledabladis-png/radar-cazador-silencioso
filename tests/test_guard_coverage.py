# -*- coding: utf-8 -*-
"""Tests del guard de cobertura (K-RUN-OUT-OF-WINDOW-01)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.guard_coverage import (
    _check_manifest,
    GUARDED_MANIFESTS,
    DEFAULT_THRESHOLD,
)


def _write_manifest(tmp_path, name, coverage=1.0, status="VALID",
                    last_date="2026-09-17", expected="2026-09-17"):
    manifest = {
        "schema_version": 1,
        "quality": {
            "coverage_pct_last": coverage,
            "status": status,
            "last_date": last_date,
            "expected_session": expected,
        },
    }
    p = tmp_path / name
    p.write_text(json.dumps(manifest), encoding="utf-8")
    return str(p)

def test_manifest_ok(tmp_path):
    path = _write_manifest(tmp_path, "ok.json")
    assert _check_manifest(path, DEFAULT_THRESHOLD) == []


def test_coverage_bajo_falla(tmp_path):
    path = _write_manifest(tmp_path, "low.json", coverage=0.5)
    reasons = _check_manifest(path, DEFAULT_THRESHOLD)
    assert len(reasons) == 1
    assert "coverage_pct_last=0.5" in reasons[0]


def test_status_invalid_falla(tmp_path):
    path = _write_manifest(tmp_path, "inv.json", status="INVALID")
    reasons = _check_manifest(path, DEFAULT_THRESHOLD)
    assert any("status=INVALID" in r for r in reasons)


def test_last_date_futuro_falla(tmp_path):
    path = _write_manifest(tmp_path, "future.json",
                           last_date="2026-09-18", expected="2026-09-17")
    reasons = _check_manifest(path, DEFAULT_THRESHOLD)
    assert any("last_date=2026-09-18 > expected_session=2026-09-17" in r
               for r in reasons)

def test_manifest_ausente_falla(tmp_path):
    reasons = _check_manifest(str(tmp_path / "missing.json"), DEFAULT_THRESHOLD)
    assert len(reasons) == 1
    assert "manifest missing" in reasons[0]


def test_manifest_json_invalido_falla(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid", encoding="utf-8")
    reasons = _check_manifest(str(p), DEFAULT_THRESHOLD)
    assert len(reasons) == 1
    assert "manifest invalid JSON" in reasons[0]


def test_borde_threshold_09499_falla(tmp_path):
    path = _write_manifest(tmp_path, "b1.json", coverage=0.9499)
    reasons = _check_manifest(path, DEFAULT_THRESHOLD)
    assert any("coverage_pct_last=0.9499" in r for r in reasons)


def test_borde_threshold_09500_pasa(tmp_path):
    path = _write_manifest(tmp_path, "b2.json", coverage=0.95)
    assert _check_manifest(path, DEFAULT_THRESHOLD) == []

def test_stale_legitimo_pasa(tmp_path):
    """STALE contractual: last_date < expected_session con coverage alto."""
    path = _write_manifest(tmp_path, "stale.json",
                           coverage=0.99, status="VALID_WITH_MISSING",
                           last_date="2026-09-16", expected="2026-09-17")
    assert _check_manifest(path, DEFAULT_THRESHOLD) == []


def test_universo_explicito():
    assert "data/stock_prices.parquet.manifest.json" in GUARDED_MANIFESTS
    assert "data/market_data.parquet.manifest.json" in GUARDED_MANIFESTS
    assert len(GUARDED_MANIFESTS) == 2