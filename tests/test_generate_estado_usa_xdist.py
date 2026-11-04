# -*- coding: utf-8 -*-
# Centinela: generate_estado_sistema.py usa pytest con -n auto y timeout.
#
# Bug 2026-11-04: sin -n auto, el script corria pytest serial sobre
# 4400+ tests y colgaba >10 min. Fix: run() acepta timeout y
# section_tests() pasa -n auto + timeout=600.
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "generate_estado_sistema.py"


def _src():
    return SCRIPT.read_text(encoding="utf-8")


def test_run_acepta_timeout():
    assert "def run(cmd, timeout=None):" in _src()


def test_run_captura_timeout_expired():
    assert "except subprocess.TimeoutExpired:" in _src()


def test_section_tests_usa_xdist():
    # --dist solo aparece con -n auto en la llamada a pytest
    assert "--dist" in _src()


def test_section_tests_timeout_600():
    assert "timeout=600" in _src()


def test_sin_pytest_serial_original():
    # El comando serial original terminaba con --no-header sin
    # timeout, seguido de cierre. Si reaparece, el bug vuelve.
    src = _src()
    marcador = "--no-header" + chr(39) + ")]"
    assert marcador not in src
