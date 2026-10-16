# -*- coding: utf-8 -*-
"""Test de contrato: todos los workflows deben usar runs-on pinneado.

Motivo: GitHub migra ubuntu-latest a Ubuntu 26 el 19-oct-2026. Sin
pin, cualquier workflow puede recibir un runner distinto sin aviso.
Con wheels binarios (numpy, pyarrow, lxml, curl_cffi), una migracion
silenciosa rompe el pip install.

Contrato: el label runs-on debe ser una version explicita
(ubuntu-24.04, ubuntu-22.04, windows-2022, etc.), nunca un alias
flotante (ubuntu-latest, windows-latest, macos-latest).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WORKFLOWS = ROOT / ".github" / "workflows"

# Alias flotantes prohibidos
FLOATING_LABELS = {
    "ubuntu-latest",
    "windows-latest",
    "macos-latest",
    "ubuntu-latest-x64",
}

# Patron de version explicita: ubuntu-XX.YY, windows-YYYY, macos-XX
EXPLICIT_RE = re.compile(
    r"^(ubuntu-\d{2}\.\d{2}|windows-\d{4}|macos-\d{2})$"
)


def _iter_workflows():
    for p in WORKFLOWS.glob("*.yml"):
        yield p
    for p in WORKFLOWS.glob("*.yaml"):
        yield p


def _extract_runs_on_lines(path):
    """Devuelve lista de (line_number, valor_runs_on)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for i, ln in enumerate(lines, 1):
        m = re.match(r"^\s*runs-on:\s*(\S+)\s*$", ln)
        if m:
            out.append((i, m.group(1).strip()))
    return out


def test_ningun_workflow_usa_alias_flotante():
    """Ningun workflow puede usar ubuntu-latest (u otro alias flotante)."""
    ofensores = []
    for wf in _iter_workflows():
        for ln, valor in _extract_runs_on_lines(wf):
            if valor in FLOATING_LABELS:
                ofensores.append(f"{wf.name}:{ln} runs-on={valor}")
    assert not ofensores, (
        "Workflows con runs-on flotante (prohibido por contrato):\n  "
        + "\n  ".join(ofensores)
    )


def test_workflows_usan_version_explicita():
    """Todos los runs-on deben ser una version explicita."""
    ofensores = []
    for wf in _iter_workflows():
        for ln, valor in _extract_runs_on_lines(wf):
            if not EXPLICIT_RE.match(valor):
                ofensores.append(f"{wf.name}:{ln} runs-on={valor}")
    assert not ofensores, (
        "Workflows con runs-on no-pinneado:\n  "
        + "\n  ".join(ofensores)
    )


def test_hay_workflows_verificados():
    """Sanity: al menos un workflow existe y tiene runs-on."""
    total = 0
    for wf in _iter_workflows():
        total += len(_extract_runs_on_lines(wf))
    assert total >= 9, f"esperaba >=9 runs-on, encontrados {total}"