"""Tests HD-4: receipt con evidencia verificable del Guard.

Dictamen auditor externo 2026-10-16: el receipt debe declarar
hechos demostrables, no literales. Casos cubiertos:
  1. VG PASS valido -> receipt escrito con guard_coverage=OK.
  2. VG passed=False -> sin receipt, exit 1.
  3. VG errors_n>0 -> sin receipt, exit 1.
  4. VG checks_n<10 -> sin receipt, exit 1.
  5. JSON ausente -> sin receipt, exit 1.
  6. JSON malformado -> sin receipt, exit 1.
  7. JSON residual (written_at antiguo) -> sin receipt, exit 1.
  8. written_at ausente -> sin receipt, exit 1.
  9. Receipt conserva schema v1 + anade campos info.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.write_completion_receipt as wcr


def _iso(hours_ago=0.0):
    t = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _manifest(target="2026-10-16", coverage=99.5, sha="deadbeef" * 8):
    return {
        "quality": {
            "expected_session": target,
            "coverage_pct_last": coverage,
        },
        "artifact": {"sha256": sha},
    }


def _vg_pass():
    return {
        "passed": True,
        "checks_n": 10,
        "errors_n": 0,
        "written_at": _iso(0.5),
    }


def _setup(tmp_path, monkeypatch, manifest, vg):
    mpath = tmp_path / "manifest.json"
    vpath = tmp_path / "vg.json"
    opath = tmp_path / "receipt.json"
    if manifest is not None:
        mpath.write_text(json.dumps(manifest), encoding="utf-8")
    if vg is not None:
        vpath.write_text(json.dumps(vg), encoding="utf-8")
    monkeypatch.setattr(wcr, "MANIFEST_PATH", mpath)
    monkeypatch.setattr(wcr, "VG_RESULT_PATH", vpath)
    monkeypatch.setattr(wcr, "OUT_PATH", opath)
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_SHA", "c0ffee" * 6)
    return opath


def test_vg_pass_writes_receipt(tmp_path, monkeypatch):
    opath = _setup(tmp_path, monkeypatch, _manifest(), _vg_pass())
    assert wcr.main() == 0
    assert opath.exists()
    r = json.loads(opath.read_text(encoding="utf-8"))
    assert r["guard_coverage"] == "OK"
    assert r["validation_gate"] == "10/10"


def test_vg_fail_no_receipt(tmp_path, monkeypatch):
    vg = _vg_pass()
    vg["passed"] = False
    opath = _setup(tmp_path, monkeypatch, _manifest(), vg)
    assert wcr.main() == 1
    assert not opath.exists()


def test_vg_errors_no_receipt(tmp_path, monkeypatch):
    vg = _vg_pass()
    vg["errors_n"] = 2
    opath = _setup(tmp_path, monkeypatch, _manifest(), vg)
    assert wcr.main() == 1
    assert not opath.exists()


def test_vg_few_checks_no_receipt(tmp_path, monkeypatch):
    vg = _vg_pass()
    vg["checks_n"] = 8
    opath = _setup(tmp_path, monkeypatch, _manifest(), vg)
    assert wcr.main() == 1
    assert not opath.exists()


def test_json_ausente_no_receipt(tmp_path, monkeypatch):
    opath = _setup(tmp_path, monkeypatch, _manifest(), None)
    assert wcr.main() == 1
    assert not opath.exists()


def test_json_malformado_no_receipt(tmp_path, monkeypatch):
    opath = _setup(tmp_path, monkeypatch, _manifest(), None)
    (tmp_path / "vg.json").write_text("{invalid json", encoding="utf-8")
    assert wcr.main() == 1
    assert not opath.exists()


def test_json_residual_no_receipt(tmp_path, monkeypatch):
    vg = _vg_pass()
    vg["written_at"] = _iso(hours_ago=24)
    opath = _setup(tmp_path, monkeypatch, _manifest(), vg)
    assert wcr.main() == 1
    assert not opath.exists()


def test_written_at_ausente_no_receipt(tmp_path, monkeypatch):
    vg = _vg_pass()
    del vg["written_at"]
    opath = _setup(tmp_path, monkeypatch, _manifest(), vg)
    assert wcr.main() == 1
    assert not opath.exists()


def test_receipt_schema_v1_compatible(tmp_path, monkeypatch):
    """El receipt emitido es valido contra schema v1 y anade campos info."""
    opath = _setup(tmp_path, monkeypatch, _manifest(), _vg_pass())
    assert wcr.main() == 0
    r = json.loads(opath.read_text(encoding="utf-8"))
    # Campos obligatorios v1
    for k in ("schema_version", "status", "target_session", "workflow",
              "run_id", "run_attempt", "completed_at", "commit_sha",
              "manifest_sha256", "manifest_coverage_pct", "guard_coverage",
              "validation_gate", "pipeline_conclusion"):
        assert k in r, "falta campo v1: {0}".format(k)
    assert r["schema_version"] == 1
    assert r["status"] == "COMPLETED"
    assert r["workflow"] == "daily_run"
    assert r["pipeline_conclusion"] == "success"
    # Campos nuevos HD-4
    assert r["success_path_reached"] is True
    assert r["validation_gate_checks_n"] == 10
    assert r["validation_gate_errors_n"] == 0
