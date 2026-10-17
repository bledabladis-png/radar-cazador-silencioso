# -*- coding: utf-8 -*-
"""Tests H19 (dictamen externo 2026-10-17): contrato de
guard_coverage_result.json.

El addendum v1.2 hizo que guard_coverage.py escribiera su resultado
a disco para que el attempt no quedara ciego. El auditor externo
(H19) exigio:
- Identidad de ejecucion (schema_version, run_id, run_attempt,
  target_session, generated_at, threshold efectivo).
- Escritura atomica (tmp + os.replace).
- Manejo explicito de excepciones.
- Placeholder al inicio, para que un guard muerto a mitad no deje
  un resultado obsoleto.
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import guard_coverage as gc


@pytest.fixture
def base_env(monkeypatch, tmp_path):
    """Env base para main(): sys.argv limpio + result path en tmp."""
    monkeypatch.setattr("sys.argv", ["guard_coverage.py"])
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    result = tmp_path / "result.json"
    monkeypatch.setattr(gc, "RESULT_PATH", result)
    return result


def _manifest_with_parquet(tmp_path, name="market", session="2026-10-16",
                            coverage=0.99, content=b"dummy"):
    manifest_p = tmp_path / f"{name}.parquet.manifest.json"
    parquet_p = tmp_path / f"{name}.parquet"
    parquet_p.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    m = {
        "artifact": {"sha256": sha},
        "quality": {
            "status": "VALID",
            "expected_session": session,
            "last_date": session,
            "coverage_pct_last": coverage,
            "by_market": {
                "US_EQUITY": {
                    "n": 255, "status": "VALID",
                    "coverage_at_session": coverage,
                }
            },
        },
    }
    manifest_p.write_text(json.dumps(m), encoding="utf-8")
    return str(manifest_p)


def test_result_contiene_identidad_ejecucion(base_env, tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "3")
    manifest_p = _manifest_with_parquet(tmp_path)
    monkeypatch.setattr(gc, "GUARDED_MANIFESTS", (manifest_p,))

    rc = gc.main()
    assert rc == 0
    d = json.loads(base_env.read_text("utf-8"))
    assert d["schema_version"] == 1
    assert d["run_id"] == 12345
    assert d["run_attempt"] == 3
    assert "generated_at" in d and d["generated_at"].endswith("Z")
    assert d["status"] == "passed"
    assert d["ok"] is True


def test_target_session_se_extrae_del_manifest(base_env, tmp_path, monkeypatch):
    manifest_p = _manifest_with_parquet(tmp_path, session="2026-10-16")
    monkeypatch.setattr(gc, "GUARDED_MANIFESTS", (manifest_p,))

    gc.main()
    d = json.loads(base_env.read_text("utf-8"))
    assert d["target_session"] == "2026-10-16"


def test_umbral_efectivo_se_respeta(base_env, monkeypatch):
    monkeypatch.setattr(gc, "GUARDED_MANIFESTS", ())
    monkeypatch.setattr("sys.argv", ["guard_coverage.py", "--threshold", "0.90"])

    gc.main()
    d = json.loads(base_env.read_text("utf-8"))
    assert d["threshold"] == pytest.approx(0.90)


def test_excepcion_en_check_manifest_se_captura(base_env, tmp_path, monkeypatch):
    manifest_p = tmp_path / "m.parquet.manifest.json"
    manifest_p.write_text("{}")

    def _boom(path, threshold):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(gc, "_check_manifest", _boom)
    monkeypatch.setattr(gc, "GUARDED_MANIFESTS", (str(manifest_p),))

    rc = gc.main()
    assert rc == 1
    d = json.loads(base_env.read_text("utf-8"))
    assert d["status"] == "failed"
    assert d["ok"] is False
    assert any("EXCEPTION RuntimeError: simulated crash" in f for f in d["failures"])


def test_escritura_es_atomica_no_queda_tmp(base_env, monkeypatch):
    monkeypatch.setattr(gc, "GUARDED_MANIFESTS", ())

    gc.main()
    tmp_file = base_env.with_name(base_env.name + ".tmp")
    assert not tmp_file.exists()
    assert base_env.exists()


def test_placeholder_running_antes_de_procesar(base_env, monkeypatch):
    estados = []
    real_replace = __import__("os").replace

    def _spy_replace(src, dst):
        try:
            d = json.loads(Path(src).read_text("utf-8"))
            estados.append(d.get("status"))
        except (OSError, ValueError):
            pass
        return real_replace(src, dst)

    monkeypatch.setattr("os.replace", _spy_replace)
    monkeypatch.setattr(gc, "GUARDED_MANIFESTS", ())

    gc.main()
    assert "running" in estados
    assert "passed" in estados
    assert estados.index("running") < estados.index("passed")


def test_failures_detail_es_estructurado(base_env, monkeypatch):
    def _fake_check(path, threshold):
        return [f"{path}: coverage_pct_last=0.5 < 0.95"]

    monkeypatch.setattr(gc, "_check_manifest", _fake_check)
    monkeypatch.setattr(gc, "GUARDED_MANIFESTS", ("data/x.manifest.json",))

    gc.main()
    d = json.loads(base_env.read_text("utf-8"))
    assert d["failures_detail"] == [
        {"manifest": "data/x.manifest.json",
         "reason": "coverage_pct_last=0.5 < 0.95"}
    ]


def test_split_failure_sin_separador():
    d = gc._split_failure("solo un mensaje sin separador")
    assert d == {"manifest": "", "reason": "solo un mensaje sin separador"}