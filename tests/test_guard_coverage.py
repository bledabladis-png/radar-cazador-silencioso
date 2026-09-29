# -*- coding: utf-8 -*-
"""Tests del guard de cobertura (K-RUN-OUT-OF-WINDOW-01).

2026-09-29: guard verifica sha256 del parquet real vs manifest declarado.
Los fixtures crean parquet sibling con sha256 correcto, y hay un test
especifico para sha256 mismatch.
"""
import hashlib
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
                    last_date="2026-09-17", expected="2026-09-17",
                    parquet_bytes=b"dummy-parquet", sha_override=None):
    """Crea parquet sibling + manifest con sha256 correcto.

    name: prefijo sin extension. Produce {name}.parquet y
          {name}.parquet.manifest.json.
    sha_override: si se pasa, se usa este sha en el manifest (para
                  simular mismatch).
    """
    parquet = tmp_path / (name + ".parquet")
    parquet.write_bytes(parquet_bytes)
    sha = sha_override or hashlib.sha256(parquet_bytes).hexdigest()
    manifest = {
        "schema_version": 1,
        "artifact": {"sha256": sha},
        "quality": {
            "coverage_pct_last": coverage,
            "status": status,
            "last_date": last_date,
            "expected_session": expected,
        },
    }
    p = tmp_path / (name + ".parquet.manifest.json")
    p.write_text(json.dumps(manifest), encoding="utf-8")
    return str(p)


def test_manifest_ok(tmp_path):
    path = _write_manifest(tmp_path, "ok")
    assert _check_manifest(path, DEFAULT_THRESHOLD) == []


def test_coverage_bajo_falla(tmp_path):
    path = _write_manifest(tmp_path, "low", coverage=0.5)
    reasons = _check_manifest(path, DEFAULT_THRESHOLD)
    assert len(reasons) == 1
    assert "coverage_pct_last=0.5" in reasons[0]


def test_status_invalid_falla(tmp_path):
    path = _write_manifest(tmp_path, "inv", status="INVALID")
    reasons = _check_manifest(path, DEFAULT_THRESHOLD)
    assert any("status=INVALID" in r for r in reasons)


def test_last_date_futuro_falla(tmp_path):
    path = _write_manifest(tmp_path, "future",
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
    path = _write_manifest(tmp_path, "b1", coverage=0.9499)
    reasons = _check_manifest(path, DEFAULT_THRESHOLD)
    assert any("coverage_pct_last=0.9499" in r for r in reasons)


def test_borde_threshold_09500_pasa(tmp_path):
    path = _write_manifest(tmp_path, "b2", coverage=0.95)
    assert _check_manifest(path, DEFAULT_THRESHOLD) == []


def test_stale_legitimo_pasa(tmp_path):
    """STALE contractual: last_date < expected_session con coverage alto."""
    path = _write_manifest(tmp_path, "stale",
                           coverage=0.99, status="VALID_WITH_MISSING",
                           last_date="2026-09-16", expected="2026-09-17")
    assert _check_manifest(path, DEFAULT_THRESHOLD) == []


def test_sha256_mismatch_falla(tmp_path):
    """2026-09-29: parquet con sha distinto al manifest -> fail-closed."""
    path = _write_manifest(tmp_path, "bad_sha", sha_override="0" * 64)
    reasons = _check_manifest(path, DEFAULT_THRESHOLD)
    assert any("sha256 mismatch" in r for r in reasons)


def test_sha256_missing_en_manifest_falla(tmp_path):
    """Manifest sin artifact.sha256 -> fail-closed."""
    parquet = tmp_path / "nosh.parquet"
    parquet.write_bytes(b"x")
    manifest = {
        "quality": {
            "coverage_pct_last": 1.0, "status": "VALID",
            "last_date": "2026-09-17", "expected_session": "2026-09-17",
        },
    }
    p = tmp_path / "nosh.parquet.manifest.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    reasons = _check_manifest(str(p), DEFAULT_THRESHOLD)
    assert any("artifact.sha256 missing" in r for r in reasons)


def test_parquet_missing_falla(tmp_path):
    """Manifest existe, parquet no -> fail-closed."""
    manifest = {
        "artifact": {"sha256": "a" * 64},
        "quality": {
            "coverage_pct_last": 1.0, "status": "VALID",
            "last_date": "2026-09-17", "expected_session": "2026-09-17",
        },
    }
    p = tmp_path / "orphan.parquet.manifest.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    reasons = _check_manifest(str(p), DEFAULT_THRESHOLD)
    assert any("parquet missing" in r for r in reasons)


def test_universo_explicito():
    assert "data/stock_prices.parquet.manifest.json" in GUARDED_MANIFESTS
    assert "data/market_data.parquet.manifest.json" in GUARDED_MANIFESTS
    assert len(GUARDED_MANIFESTS) == 2