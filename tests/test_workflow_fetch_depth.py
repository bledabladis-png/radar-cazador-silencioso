# -*- coding: utf-8 -*-
"""Valida que todo job con pytest tiene checkout con fetch-depth: 0.

Regresion 2026-10-17: actions/checkout@v4 hace shallow clone por
defecto (fetch-depth: 1). El test test_gobernanza.py::
test_registro_real_pasa verifica existencia de los SHAs referenciados
en REGISTRO.yaml contra git. En CI solo existia HEAD, los 37 SHAs
"no existian", el test fallaba y el run 38993972583 murio en
"Run tests" con exit 1.

Este test barre .github/workflows/*.yml, encuentra jobs que ejecutan
pytest en algun step, y exige que el primer checkout del job tenga
with.fetch-depth: 0. Coste ~0.1s.
"""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = ROOT / ".github" / "workflows"


def _check_workflow(wf_dict, wf_name):
    """Devuelve lista de errores (strings). Vacia si OK."""
    errors = []
    jobs = wf_dict.get("jobs") or {}
    for job_name, job in jobs.items():
        steps = job.get("steps") or []
        has_pytest = any("pytest" in (s.get("run") or "") for s in steps)
        if not has_pytest:
            continue
        checkouts = [
            s for s in steps
            if "actions/checkout@" in (s.get("uses") or "")
        ]
        if not checkouts:
            errors.append(
                "{0}::{1}: corre pytest sin actions/checkout".format(
                    wf_name, job_name
                )
            )
            continue
        first_co = checkouts[0]
        fd = (first_co.get("with") or {}).get("fetch-depth")
        if fd != 0:
            errors.append(
                "{0}::{1}: corre pytest pero su checkout tiene "
                "fetch-depth={2!r} (debe ser 0)".format(
                    wf_name, job_name, fd
                )
            )
    return errors


def test_workflows_con_pytest_usan_fetch_depth_0():
    """Ningun job con pytest queda con shallow clone por defecto."""
    errors = []
    for wf in sorted(WORKFLOWS_DIR.glob("*.yml")):
        d = yaml.safe_load(wf.read_text(encoding="utf-8"))
        errors.extend(_check_workflow(d, wf.name))
    if errors:
        pytest.fail(
            "Workflows con pytest sin fetch-depth: 0:\n"
            + "\n".join("  " + e for e in errors)
        )


def test_detecta_workflow_sin_fetch_depth():
    """Negativo: checkout sin fetch-depth + pytest -> error."""
    bad = {
        "jobs": {
            "test": {
                "steps": [
                    {"uses": "actions/checkout@v4"},
                    {"run": "pytest tests/"},
                ]
            }
        }
    }
    errors = _check_workflow(bad, "bad.yml")
    assert len(errors) == 1
    assert "fetch-depth" in errors[0]


def test_detecta_workflow_sin_checkout():
    """Negativo: pytest sin checkout -> error."""
    bad = {
        "jobs": {
            "test": {
                "steps": [
                    {"run": "pytest tests/"},
                ]
            }
        }
    }
    errors = _check_workflow(bad, "bad.yml")
    assert len(errors) == 1
    assert "sin actions/checkout" in errors[0]


def test_workflow_sin_pytest_no_requiere_fetch_depth():
    """Jobs sin pytest no exigen fetch-depth: 0."""
    ok = {
        "jobs": {
            "test": {
                "steps": [
                    {"uses": "actions/checkout@v4"},
                    {"run": "echo hola"},
                ]
            }
        }
    }
    assert _check_workflow(ok, "ok.yml") == []


def test_workflow_con_fetch_depth_correcto_pasa():
    """Job con pytest y fetch-depth: 0 -> sin errores."""
    ok = {
        "jobs": {
            "test": {
                "steps": [
                    {"uses": "actions/checkout@v4",
                     "with": {"fetch-depth": 0}},
                    {"run": "pytest tests/"},
                ]
            }
        }
    }
    assert _check_workflow(ok, "ok.yml") == []