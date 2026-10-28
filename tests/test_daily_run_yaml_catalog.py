"""Contrato: daily_run.yml invoca los steps IAE que materializan
catalog_assignments.csv y catalog_membership.csv.

Contexto (2026-10-28): el run scheduled ejecutaba
regenerate_radar_catalog.py (actualiza manifest + snapshots) pero
nunca build_catalog_csvs.py (materializa los 2 CSV derivados).
Resultado: manifest con snapshot 2026-10-28_01, assignments en 255,
membership en 739. Los tests de integridad lo detectaron;
test_membership_versiones_completas fallaba con
"faltan: {2026-10-28_01}".
"""
from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = ROOT / ".github" / "workflows" / "daily_run.yml"


def _steps():
    with YAML_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # localizar el job run-system
    jobs = data.get("jobs") or {}
    for name, job in jobs.items():
        if name == "run-system":
            return job.get("steps") or []
    raise AssertionError("job run-system no encontrado en daily_run.yml")


def _step_names():
    return [s.get("name", "") for s in _steps()]


def _step_runs():
    return [s.get("run", "") for s in _steps() if s.get("run")]


def test_step_build_catalog_csvs_existe():
    assert any("build_catalog_csvs" in (s.get("run") or "")
               for s in _steps()), (
        "daily_run.yml no invoca scripts/build_catalog_csvs.py. "
        "El manifest avanzaria pero los CSV derivados quedarian "
        "desincronizados (bug 2026-10-28)."
    )


def test_step_build_catalog_csvs_despues_de_regenerate():
    """El orden importa: primero manifest + snapshots, luego CSV."""
    runs = _step_runs()
    idx_regen = next(
        (i for i, r in enumerate(runs)
         if "regenerate_radar_catalog.py" in r), None)
    idx_build = next(
        (i for i, r in enumerate(runs)
         if "build_catalog_csvs.py" in r), None)
    assert idx_regen is not None, "falta step regenerate_radar_catalog.py"
    assert idx_build is not None, "falta step build_catalog_csvs.py"
    assert idx_build > idx_regen, (
        "build_catalog_csvs.py debe ejecutarse DESPUES de "
        "regenerate_radar_catalog.py (consume su output)."
    )


def test_step_build_catalog_csvs_no_enmascara_exit_code():
    """Si el step falla, el pipeline debe parar (igual que regenerate)."""
    for s in _steps():
        run = s.get("run") or ""
        if "build_catalog_csvs.py" not in run:
            continue
        assert s.get("continue-on-error") not in (True, "true"), (
            "build_catalog_csvs.py no debe tener continue-on-error: "
            "si falla, debe propagar el exit code."
        )
        return
    raise AssertionError("step build_catalog_csvs.py no encontrado")