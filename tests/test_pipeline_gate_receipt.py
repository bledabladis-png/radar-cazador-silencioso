# -*- coding: utf-8 -*-
"""Tests del Completion Receipt (contrato v1, invariantes I1-I8).

Referencia: docs/auditoria/daily_run_gate_contrato_v1.md
Cada test verifica una invariante del contrato. Sin excepciones.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.pipeline_gate as gate


@pytest.fixture
def isolated_project(tmp_path, monkeypatch):
    """PROJECT_ROOT apunta a tmp. Sin GH_TOKEN -> _github_api devuelve None."""
    monkeypatch.setattr(gate, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    return tmp_path


def _make_download_df(target_session, coverage_frac, tickers=gate.GATE_PANEL_USA):
    """Simula yf.download con MultiIndex (Close, ticker)."""
    n = len(tickers)
    n_valid = int(round(n * coverage_frac))
    vals = [1.0] * n_valid + [float("nan")] * (n - n_valid)
    return pd.DataFrame(
        [vals],
        index=pd.DatetimeIndex([pd.Timestamp(target_session)]),
        columns=pd.MultiIndex.from_product([["Close"], list(tickers)]),
    )


def _valid_receipt(target_session="2026-09-30"):
    return {
        "schema_version": 1,
        "status": "COMPLETED",
        "target_session": target_session,
        "workflow": "daily_run",
        "run_id": 36876380805,
        "run_attempt": 1,
        "completed_at": "2026-10-01T14:45:12Z",
        "commit_sha": "abc" * 10,
        "manifest_sha256": "def" * 10,
        "manifest_coverage_pct": 99.12,
        "guard_coverage": "OK",
        "validation_gate": "10/10",
        "pipeline_conclusion": "success",
    }


# ---------------- I1: sin receipt, sin parquet -> no CURRENT ----------------

def test_i1_sin_receipt_sin_parquet_no_current(isolated_project):
    """I1: nunca CURRENT por ausencia de parquet.

    No hay receipt. No hay parquet en CI. El gate cae al probe.
    """
    df = _make_download_df("2026-09-30", coverage_frac=1.0)
    with patch.object(gate.yf, "download", return_value=df):
        result = gate.evaluate("2026-09-30")
    assert result["state"] != "CURRENT"
    assert result["state"] == "READY"


# ---------------- I2: manifest aislado no basta ----------------

def test_i2_manifest_valido_sin_receipt_no_current(isolated_project):
    """I2: nunca CURRENT por un manifest aislado.

    Aunque _manifest_satisfies pueda dar True, evaluate no lo consulta
    para CURRENT. Solo el receipt.
    """
    # Escribir manifest "valido" (aunque no hay parquet sibling en CI)
    manifest_dir = isolated_project / "data"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "stock_prices.parquet.manifest.json").write_text(json.dumps({
        "artifact": {"sha256": "0" * 64},
        "quality": {
            "expected_session": "2026-09-30",
            "coverage_pct_last": 0.99,
            "status": "VALID",
        },
    }), encoding="utf-8")

    df = _make_download_df("2026-09-30", coverage_frac=1.0)
    with patch.object(gate.yf, "download", return_value=df), \
         patch.object(gate, "find_completion_receipt", return_value=None):
        result = gate.evaluate("2026-09-30")
    assert result["state"] == "READY"
    assert result["state"] != "CURRENT"


# ---------------- I3: receipt solo tras ultimo check ----------------

def test_i3_step_receipt_con_if_success():
    """I3: el step 'Upload completion receipt' tiene if: success().

    Verificacion estatica del yml. El step debe existir, tener if:
    success() y estar tras guard_coverage y Commit and push.
    """
    yml = (ROOT / ".github" / "workflows" / "daily_run.yml").read_text(
        encoding="utf-8"
    )
    assert "Upload completion receipt" in yml, (
        "El step 'Upload completion receipt' debe existir en daily_run.yml"
    )
    idx_upload = yml.find("Upload completion receipt")
    idx_guard = yml.find("Guard coverage")
    idx_commit = yml.find("Commit and push hist/state")
    assert idx_guard != -1, "Guard coverage debe existir"
    assert idx_commit != -1, "Commit and push debe existir"
    assert idx_upload > idx_guard, "Receipt debe ir despues de guard_coverage"
    assert idx_upload > idx_commit, "Receipt debe ir despues del push"
    # if: success() en las lineas cercanas al step
    fragment = yml[idx_upload - 200:idx_upload + 400]
    assert "if: success()" in fragment, (
        "El step del receipt debe tener if: success()"
    )


# ---------------- I4: schema del receipt ----------------

def test_i4_validate_schema_campos_obligatorios():
    """I4: el schema del receipt valida todos los campos."""
    r = _valid_receipt("2026-09-30")
    assert gate._validate_receipt_schema(r, "2026-09-30") is True

    # Campo faltante -> False
    for campo in ("schema_version", "status", "target_session",
                  "workflow", "run_id", "validation_gate",
                  "pipeline_conclusion", "completed_at"):
        r_bad = dict(r)
        del r_bad[campo]
        assert gate._validate_receipt_schema(r_bad, "2026-09-30") is False, (
            "Falta {0}: no debe validar".format(campo)
        )

    # schema_version incorrecto -> False
    r_v2 = dict(r, schema_version=2)
    assert gate._validate_receipt_schema(r_v2, "2026-09-30") is False

    # status incorrecto -> False
    r_st = dict(r, status="PENDING")
    assert gate._validate_receipt_schema(r_st, "2026-09-30") is False

    # validation_gate incorrecto -> False
    r_g = dict(r, validation_gate="9/10")
    assert gate._validate_receipt_schema(r_g, "2026-09-30") is False

    # pipeline_conclusion incorrecto -> False
    r_c = dict(r, pipeline_conclusion="failure")
    assert gate._validate_receipt_schema(r_c, "2026-09-30") is False


# ---------------- I5: receipt expirado es fail-open ----------------

def test_i5_receipt_expirado_no_current(isolated_project, monkeypatch):
    """I5: si el receipt expira (artifact ausente), el gate cae al probe.

    Simulamos: API devuelve lista vacia de artifacts.
    """
    def _api_empty(path, params=None):
        if path == "actions/artifacts":
            return {"artifacts": []}
        return None

    df = _make_download_df("2026-09-30", coverage_frac=1.0)
    with patch.object(gate, "_github_api", side_effect=_api_empty), \
         patch.object(gate.yf, "download", return_value=df):
        result = gate.evaluate("2026-09-30")
    assert result["state"] != "CURRENT"
    assert result["state"] == "READY"


# ---------------- I6: API failure != CURRENT ----------------

def test_i6_api_error_no_current(isolated_project):
    """I6: cualquier error de API -> None -> sin CURRENT."""
    def _api_error(path, params=None):
        raise RuntimeError("network down")

    df = _make_download_df("2026-09-30", coverage_frac=1.0)
    with patch.object(gate, "_github_api", side_effect=_api_error), \
         patch.object(gate.yf, "download", return_value=df):
        result = gate.evaluate("2026-09-30")
    assert result["state"] != "CURRENT"
    assert result["state"] == "READY"


def test_i6_sin_token_no_current(isolated_project):
    """I6 (variante): sin GH_TOKEN, _github_api no intenta la llamada."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    # Sin token, requests.get no se llama. Simulamos con patch.
    def _api_no_token(path, params=None):
        return None

    df = _make_download_df("2026-09-30", coverage_frac=1.0)
    with patch.object(gate, "_github_api", side_effect=_api_no_token), \
         patch.object(gate.yf, "download", return_value=df):
        result = gate.evaluate("2026-09-30")
    assert result["state"] == "READY"
    monkeypatch.undo()


# ---------------- I7: target_session exacta ----------------

def test_i7_receipt_de_otra_session_no_current(isolated_project):
    """I7: receipt para 2026-09-29 no sirve para 2026-09-30."""
    r = _valid_receipt("2026-09-29")
    # El schema check falla porque target_session no coincide
    assert gate._validate_receipt_schema(r, "2026-09-30") is False

    # Simulamos: find_completion_receipt devuelve None (schema invalido
    # internamente). Evaluamos 2026-09-30.
    df = _make_download_df("2026-09-30", coverage_frac=1.0)
    with patch.object(gate, "find_completion_receipt", return_value=None), \
         patch.object(gate.yf, "download", return_value=df):
        result = gate.evaluate("2026-09-30")
    assert result["state"] == "READY"


# ---------------- I8: receipt no sustituye manifest ----------------

def test_i8_manifest_satisfies_sigue_fail_closed(isolated_project):
    """I8: _manifest_satisfies conserva su contrato original.

    Sin parquet sibling, devuelve False aunque el manifest este bien.
    """
    manifest = {
        "artifact": {"sha256": "0" * 64},
        "quality": {
            "expected_session": "2026-09-30",
            "coverage_pct_last": 0.99,
        },
    }
    # No creamos parquet sibling. Debe devolver False.
    assert gate._manifest_satisfies(manifest, "2026-09-30") is False

    # Con parquet sibling, y sha correcto -> True
    parquet = isolated_project / "data" / "stock_prices.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    import hashlib as _h
    content = b"dummy"
    parquet.write_bytes(content)
    manifest_ok = {
        "artifact": {"sha256": _h.sha256(content).hexdigest()},
        "quality": {
            "expected_session": "2026-09-30",
            "coverage_pct_last": 0.99,
        },
    }
    assert gate._manifest_satisfies(manifest_ok, "2026-09-30") is True


# ---------------- Test del schema helper ----------------

def test_validate_receipt_schema_target_session_mismatch():
    r = _valid_receipt("2026-09-30")
    assert gate._validate_receipt_schema(r, "2026-09-29") is False
    assert gate._validate_receipt_schema(r, "2026-09-30") is True
