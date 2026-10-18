"""Fail-closed de generate_estado_sistema.py.

Garantiza que main() retorna int != 0 cuando pytest, compileall o
pyflakes fallan, y que el snapshot se escribe igualmente con el
Exit code real (fuente autoritativa nunca stale).

Tests hermeticos: monkeypatch de run + rutas a tmp_path. No leen
data/ ni outputs/.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import generate_estado_sistema as ges


def _fake_run(pytest_rc=0, compileall_rc=0, pyflakes_rc=0, pytest_out=None, pyflakes_out=""):
    if pytest_out is None:
        pytest_out = "3739 passed, 5 skipped in 100s"
    def _run(cmd):
        if 'pytest' in cmd:
            return pytest_out, pytest_rc
        if 'compileall' in cmd:
            return "", compileall_rc
        if 'pyflakes' in cmd:
            return pyflakes_out, pyflakes_rc
        return "", 0
    return _run


def _setup_hermetic(monkeypatch, tmp_path, fake_run):
    monkeypatch.setattr(ges, 'run', fake_run)
    monkeypatch.setattr(ges, 'ROOT', tmp_path)
    monkeypatch.setattr(ges, 'IAE_ROOT', tmp_path / "iae_no_existe")
    monkeypatch.setattr(ges, 'MAPPINGS_IAE', [])
    monkeypatch.setattr(ges, 'SCRIPTS_IAE', [])
    monkeypatch.setattr(ges, 'EVIDENCE_ROOT', tmp_path / "evid_no_existe")
    monkeypatch.setattr(ges, 'OUT', tmp_path / "ESTADO.md")


def test_main_retorna_0_si_todo_ok(monkeypatch, tmp_path):
    _setup_hermetic(monkeypatch, tmp_path, _fake_run())
    rc = ges.main()
    assert rc == 0
    content = (tmp_path / "ESTADO.md").read_text(encoding="utf-8")
    assert '- **Exit code:** 0' in content


def test_main_retorna_1_si_pytest_falla(monkeypatch, tmp_path):
    out = "1 failed, 3738 passed, 5 skipped in 100s\nFAILED tests/foo.py::test_bar - assert"
    _setup_hermetic(monkeypatch, tmp_path, _fake_run(pytest_rc=1, pytest_out=out))
    rc = ges.main()
    assert rc == 1
    content = (tmp_path / "ESTADO.md").read_text(encoding="utf-8")
    assert '- **Exit code:** 1' in content
    assert 'FAILED tests/foo.py::test_bar - assert' in content


def test_main_retorna_1_si_compileall_falla(monkeypatch, tmp_path):
    _setup_hermetic(monkeypatch, tmp_path, _fake_run(compileall_rc=1))
    assert ges.main() == 1


def test_main_retorna_1_si_pyflakes_falla(monkeypatch, tmp_path):
    _setup_hermetic(monkeypatch, tmp_path, _fake_run(pyflakes_rc=1, pyflakes_out="warning: x"))
    assert ges.main() == 1