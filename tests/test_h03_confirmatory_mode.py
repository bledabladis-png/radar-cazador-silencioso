"""Tests H-03 dictamen doc 76: --confirmatory en capture_experiment_env.

Llama main() in-process (sin subprocess) para no depender de --out-dir
inexistente. Monkeypatch sobre el modulo para aislar el output.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def ce_mod(monkeypatch, tmp_path):
    """Importa el modulo y monkeypatcha OUT_DIR."""
    import scripts.audit.capture_experiment_env as ce
    out_dir = tmp_path / "experiments"
    monkeypatch.setattr(ce, "OUT_DIR", out_dir)
    return ce, out_dir


def _run(ce, argv):
    """Ejecuta main(argv), captura stdout, retorna (rc, stdout, out_path)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = ce.main(argv)
    out_path = ce.OUT_DIR / (argv[argv.index("--experiment-id") + 1] + ".json")
    return rc, buf.getvalue(), out_path


def test_mutex_confirmatory_y_allow_dirty(ce_mod):
    ce, out_dir = ce_mod
    rc, stdout, out_path = _run(ce, [
        "--experiment-id", "test_mutex1",
        "--candidate", "C1-M5",
        "--confirmatory",
        "--allow-dirty",
    ])
    assert rc == 1
    assert "incompatible" in stdout.lower()
    # El manifest se escribe igualmente con errors[] poblado
    assert out_path.exists()
    d = json.loads(out_path.read_text(encoding="utf-8"))
    assert any("incompatible" in e.lower() for e in d["errors"])


def test_mutex_confirmatory_y_allow_missing_holdout(ce_mod):
    ce, out_dir = ce_mod
    rc, stdout, out_path = _run(ce, [
        "--experiment-id", "test_mutex2",
        "--candidate", "C1-M5",
        "--confirmatory",
        "--allow-missing-holdout",
    ])
    assert rc == 1
    assert "incompatible" in stdout.lower()


def test_modo_exploratorio_por_defecto(ce_mod):
    ce, out_dir = ce_mod
    rc, stdout, out_path = _run(ce, [
        "--experiment-id", "test_explor",
        "--candidate", "C1-M5",
        "--allow-dirty",
        "--allow-missing-holdout",
    ])
    assert out_path.exists()
    d = json.loads(out_path.read_text(encoding="utf-8"))
    assert d.get("mode") == "exploratory"
    assert d.get("holdout_verify", {}).get("checked") is False
    assert d.get("pip_freeze") == []


def test_confirmatory_sobre_repo_dirty_falla(ce_mod, monkeypatch):
    """En repo dirty, --confirmatory debe fallar con mensaje claro.

    Monkeypatch de _git_is_dirty para inyectar el estado dirty sin
    depender del working tree real. El test original hacia skip en CI
    (arbol siempre limpio) y nunca ejercia el camino bajo prueba.
    """
    ce, out_dir = ce_mod
    monkeypatch.setattr(ce, "_git_is_dirty", lambda: True)
    rc, stdout, out_path = _run(ce, [
        "--experiment-id", "test_dirty",
        "--candidate", "C1-M5",
        "--confirmatory",
    ])
    assert rc == 1
    assert out_path.exists()
    d = json.loads(out_path.read_text(encoding="utf-8"))
    assert d.get("mode") == "confirmatory"
    assert any("dirty" in e.lower() or "confirmatorio" in e.lower()
               for e in d["errors"])
