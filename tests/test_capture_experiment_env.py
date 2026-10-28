"""Tests de capture_experiment_env.py (H-02 SOW-HOLDOUT-IMMUTABLE).

Cubren las precondiciones que el dictamen 2026-10-28 exige:
- El commit del pre-registro debe ser ancestro de HEAD (evidencia
  temporal: pre-registro precede a la ejecucion).
- El working tree debe estar limpio (un experimento confirmatorio
  no se captura sobre cambios sin commitear).
- El holdout_manifest.json debe existir (el experimento queda atado
  a un contenido concreto del holdout).
- El manifest incluye requirements + validators + modules + params +
  holdout.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.audit import capture_experiment_env as cap


# --- helpers ---

def _git(*args):
    return subprocess.run(
        ["git"] + list(args), cwd=str(ROOT),
        capture_output=True, text=True, timeout=15,
    )


def _head():
    r = _git("rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else None


def _ancestor_of_head():
    """Commit ancestro de HEAD (para prereg valido)."""
    r = _git("rev-parse", "HEAD~1")
    return r.stdout.strip() if r.returncode == 0 else None


def _commit_exists():
    return _head() is not None


# --- H-02: precondiciones ---

def test_precondicion_falla_si_dirty(tmp_path, monkeypatch):
    """Working tree dirty sin --allow-dirty -> exit 1 (fail-close)."""
    if not _commit_exists():
        pytest.skip("no git / sin commit")
    # Forzar dirty via monkeypatch de _git_is_dirty (evita ensuciar el repo real)
    monkeypatch.setattr(cap, "OUT_DIR", tmp_path)
    monkeypatch.setattr(cap, "_git_is_dirty", lambda: True)
    rc = cap.main([
        "--experiment-id", "test_dirty",
        "--candidate", "test",
    ])
    assert rc == 1, "dirty deberia fallar sin --allow-dirty"


def test_precondicion_falla_si_prereg_no_ancestro(tmp_path, monkeypatch):
    """prereg-commit no ancestro -> exit 1 (evidencia temporal invalida)."""
    if not _commit_exists():
        pytest.skip("no git / sin commit")
    monkeypatch.setattr(cap, "OUT_DIR", tmp_path)
    monkeypatch.setattr(cap, "_git_is_dirty", lambda: False)
    monkeypatch.setattr(cap, "_is_ancestor", lambda c: False)
    monkeypatch.setattr(cap, "_git_commit_exists", lambda c: True)
    rc = cap.main([
        "--experiment-id", "test_prereg_no_ancestro",
        "--candidate", "test",
        "--prereg-commit", "deadbeef",
    ])
    assert rc == 1


def test_precondicion_falla_si_prereg_no_existe(tmp_path, monkeypatch):
    """prereg-commit inexistente -> exit 1."""
    if not _commit_exists():
        pytest.skip("no git / sin commit")
    monkeypatch.setattr(cap, "OUT_DIR", tmp_path)
    monkeypatch.setattr(cap, "_git_is_dirty", lambda: False)
    monkeypatch.setattr(cap, "_git_commit_exists", lambda c: False)
    rc = cap.main([
        "--experiment-id", "test_prereg_no_existe",
        "--candidate", "test",
        "--prereg-commit", "0" * 40,
    ])
    assert rc == 1


def test_precondicion_falla_si_no_holdout_manifest(tmp_path, monkeypatch):
    """Sin holdout_manifest.json -> exit 1 (salvo --allow-missing-holdout)."""
    if not _commit_exists():
        pytest.skip("no git / sin commit")
    monkeypatch.setattr(cap, "OUT_DIR", tmp_path)
    monkeypatch.setattr(cap, "HOLDOUT_MANIFEST", tmp_path / "no_existe.json")
    monkeypatch.setattr(cap, "_git_is_dirty", lambda: False)
    rc = cap.main([
        "--experiment-id", "test_no_holdout",
        "--candidate", "test",
    ])
    assert rc == 1


# --- H-02: manifest escrito cuando todo OK ---

def test_manifest_ok_con_allow_dirty(tmp_path, monkeypatch):
    """Con --allow-dirty y holdout presente, escribe manifest."""
    if not _commit_exists():
        pytest.skip("no git / sin commit")
    if not cap.HOLDOUT_MANIFEST.exists():
        pytest.skip("holdout_manifest.json no disponible")
    monkeypatch.setattr(cap, "OUT_DIR", tmp_path)
    rc = cap.main([
        "--experiment-id", "test_ok",
        "--candidate", "test",
        "--allow-dirty",
        "--seed", "42",
    ])
    assert rc == 0, "con --allow-dirty deberia pasar"
    out = tmp_path / "test_ok.json"
    assert out.exists(), "manifest no escrito"
    d = json.loads(out.read_text(encoding="utf-8"))
    # Campos obligatorios H-02
    for k in (
        "schema_version", "experiment_id", "candidate", "captured_at",
        "git", "prereg", "python", "requirements", "validators",
        "validator_params", "modules", "sow_params", "seed", "holdout",
        "warnings", "errors",
    ):
        assert k in d, "falta " + k
    assert d["experiment_id"] == "test_ok"
    assert d["candidate"] == "test"
    assert d["seed"] == 42
    # Git info
    assert d["git"]["head_sha"]
    assert d["git"]["branch"] == "main"
    # Requirements
    assert d["requirements"]["exists"] is True
    assert d["requirements"]["sha256"]
    assert d["requirements"]["pinned_count"] > 0
    # Validators: los 5 purged_kfold_sow_*.py
    assert len(d["validators"]) >= 1
    # Modules: los 6 wyckoff_*.py
    assert len(d["modules"]) >= 1
    # sow_params: WYCKOFF_SOW_* de config.settings
    assert "WYCKOFF_SOW_WINDOW_N" in d["sow_params"]
    assert "WYCKOFF_SOW_MAX_AGE_M" in d["sow_params"]
    # holdout: cohort + r3000 con sha256
    assert d["holdout"]["exists"] is True
    assert "cohort" in d["holdout"]
    assert "r3000" in d["holdout"]
    assert d["holdout"]["cohort"]["sha256"]
    assert d["holdout"]["cohort"]["n_tickers"] == 268
    assert d["holdout"]["r3000"]["n_tickers"] == 3423


def test_manifest_incluye_candidate_de_cada_validador(tmp_path, monkeypatch):
    """Cada validador tiene CANDIDATE con N, M, X_ATR, Y_VOL."""
    if not _commit_exists():
        pytest.skip("no git / sin commit")
    if not cap.HOLDOUT_MANIFEST.exists():
        pytest.skip("holdout_manifest.json no disponible")
    monkeypatch.setattr(cap, "OUT_DIR", tmp_path)
    rc = cap.main([
        "--experiment-id", "test_candidates",
        "--candidate", "test",
        "--allow-dirty",
    ])
    assert rc == 0
    d = json.loads((tmp_path / "test_candidates.json").read_text(encoding="utf-8"))
    vps = d["validator_params"]
    assert len(vps) >= 1
    for vrel, vp in vps.items():
        assert isinstance(vp, dict)
        assert "CANDIDATE" in vp, vrel + " sin CANDIDATE"
        c = vp["CANDIDATE"]
        for key in ("N", "M", "X_ATR", "Y_VOL"):
            assert key in c, vrel + " CANDIDATE sin " + key
        assert "EMBARGO" in vp, vrel + " sin EMBARGO"
        assert "H" in vp, vrel + " sin H"


def test_precondicion_ok_con_prereg_ancestro(tmp_path, monkeypatch):
    """Con prereg ancestro real, exit 0."""
    if not _commit_exists():
        pytest.skip("no git / sin commit")
    if not cap.HOLDOUT_MANIFEST.exists():
        pytest.skip("holdout_manifest.json no disponible")
    ancestor = _ancestor_of_head()
    if not ancestor:
        pytest.skip("no hay commits anteriores para ancestro")
    monkeypatch.setattr(cap, "OUT_DIR", tmp_path)
    rc = cap.main([
        "--experiment-id", "test_prereg_ancestro",
        "--candidate", "test",
        "--prereg-commit", ancestor,
        "--allow-dirty",
    ])
    assert rc == 0
    d = json.loads((tmp_path / "test_prereg_ancestro.json").read_text(encoding="utf-8"))
    assert d["prereg"]["checked"] is True
    assert d["prereg"]["exists"] is True
    assert d["prereg"]["is_ancestor_of_head"] is True