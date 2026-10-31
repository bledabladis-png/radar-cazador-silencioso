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

# --- Addendum v1.2 (2026-10-17): _write_result persistido ---

def test_guard_escribe_result_ok(tmp_path, monkeypatch):
    """Guard OK: escribe outputs/state/guard_coverage_result.json."""
    import json
    import scripts.guard_coverage as gc

    result_path = tmp_path / "outputs" / "state" / "guard_coverage_result.json"
    monkeypatch.setattr(gc, "RESULT_PATH", result_path)
    monkeypatch.setattr(gc, "GUARDED_MANIFESTS", ())

    monkeypatch.setattr("sys.argv", ["guard_coverage.py"])
    rc = gc.main()
    assert rc == 0
    assert result_path.exists()
    d = json.loads(result_path.read_text(encoding="utf-8"))
    assert d["ok"] is True
    assert d["failures"] == []
    assert "threshold" in d


def test_guard_escribe_result_fallo(tmp_path, monkeypatch):
    """Guard FAIL: escribe ok=False y las failures."""
    import json
    import scripts.guard_coverage as gc

    result_path = tmp_path / "outputs" / "state" / "guard_coverage_result.json"
    monkeypatch.setattr(gc, "RESULT_PATH", result_path)

    def _fake_check(path, threshold):
        return [path + ": coverage_pct_last=0.927 < 0.95"]

    monkeypatch.setattr(gc, "_check_manifest", _fake_check)
    monkeypatch.setattr(gc, "GUARDED_MANIFESTS", ("data/fake.manifest.json",))
    monkeypatch.setattr("sys.argv", ["guard_coverage.py"])

    rc = gc.main()
    assert rc == 1
    d = json.loads(result_path.read_text(encoding="utf-8"))
    assert d["ok"] is False
    assert any("0.927" in f for f in d["failures"])


# --- GUARD-COVERAGE-LAST-DATE-PARCIAL (2026-10-31) ---

def _write_manifest_eff(tmp_path, name,
                        coverage_last=1.0, coverage_eff=None,
                        effective_session=None,
                        last_date="2026-09-17", expected="2026-09-17",
                        status="VALID", parquet_bytes=b"dummy-parquet"):
    """Variante que anade coverage_pct_effective y effective_session."""
    parquet = tmp_path / (name + ".parquet")
    parquet.write_bytes(parquet_bytes)
    sha = hashlib.sha256(parquet_bytes).hexdigest()
    quality = {
        "coverage_pct_last": coverage_last,
        "status": status,
        "last_date": last_date,
        "expected_session": expected,
    }
    if coverage_eff is not None:
        quality["coverage_pct_effective"] = coverage_eff
    if effective_session is not None:
        quality["effective_session"] = effective_session
    manifest = {
        "schema_version": 1,
        "artifact": {"sha256": sha},
        "quality": quality,
    }
    p = tmp_path / (name + ".parquet.manifest.json")
    p.write_text(json.dumps(manifest), encoding="utf-8")
    return str(p)


def test_guard_usa_effective_sobre_last(tmp_path):
    """coverage_pct_effective=0.99 gana a coverage_pct_last=0.5."""
    path = _write_manifest_eff(
        tmp_path, "eff_ok",
        coverage_last=0.5, coverage_eff=0.99,
        effective_session="2026-09-16",
    )
    reasons = _check_manifest(path, DEFAULT_THRESHOLD)
    assert reasons == [], (
        "El guard debe preferir coverage_pct_effective=0.99 sobre "
        "coverage_pct_last=0.5. Razon detectada: {0}".format(reasons))


def test_guard_fallback_last_si_no_effective(tmp_path):
    """Sin coverage_pct_effective, guard usa coverage_pct_last (backward compat)."""
    path = _write_manifest_eff(
        tmp_path, "no_eff",
        coverage_last=0.5, coverage_eff=None,
    )
    reasons = _check_manifest(path, DEFAULT_THRESHOLD)
    assert len(reasons) == 1
    assert "coverage_pct_last=0.5" in reasons[0]
