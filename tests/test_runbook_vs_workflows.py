# -*- coding: utf-8 -*-
"""Centinela: los crons documentados en 07_RUNBOOK coinciden con los YAML.

Regresion 2026-11-03: el runbook decia que update_sec_nport era
trimestral (dia 20), pero el YAML ya era semanal (miercoles) desde
2026-10-16 (dictamen H5). Sin centinela, la documentacion puede
divergir del codigo sin que ningun test lo detecte.

Alcance: los 7 workflows "update_*" (donde el runbook lista cron
explicito). daily_run (5 slots) y health_check (unico slot) se
gestionan aparte.
"""
from __future__ import annotations

import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "Consolidacion_Documentos" / "07_RUNBOOK.md"
WF_DIR = ROOT / ".github" / "workflows"

# Formato del runbook: "    update_X.yml    <cron>    CEST HH:MM"
RUNBOOK_LINE_RE = re.compile(
    r"^\s+(update_[a-z0-9_]+\.yml)\s+(\S.*?)\s{2,}CEST\s+\d{1,2}:\d{2}\s*$",
    re.MULTILINE,
)


def _parse_runbook_crons() -> dict[str, str]:
    """Devuelve {workflow_name: cron_str} segun el runbook."""
    text = RUNBOOK.read_text(encoding="utf-8-sig")
    out = {}
    for m in RUNBOOK_LINE_RE.finditer(text):
        wf = m.group(1).strip()
        cron = m.group(2).strip()
        out[wf] = cron
    return out


def _parse_yaml_crons(path: pathlib.Path) -> list[str]:
    """Devuelve la lista de crons del workflow (0..N)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        return []
    # YAML 1.1: PyYAML interpreta la clave top-level 'on:' como
    # boolean True. Se busca primero 'on' (str) y luego True.
    on = data.get("on")
    if on is None:
        on = data.get(True)
    if not isinstance(on, dict):
        return []
    schedule = on.get("schedule")
    if not isinstance(schedule, list):
        return []
    return [str(s.get("cron", "")).strip() for s in schedule if isinstance(s, dict) and s.get("cron")]


def test_runbook_crons_coinciden_con_yaml():
    """Cada cron documentado en el runbook debe coincidir con el YAML real."""
    runbook = _parse_runbook_crons()
    assert runbook, "el runbook no expone ningun cron en el formato esperado"

    errores = []
    for wf_name, cron_doc in runbook.items():
        wf_path = WF_DIR / wf_name
        if not wf_path.exists():
            errores.append(f"{wf_name}: referenciado en runbook pero no existe en .github/workflows/")
            continue
        yaml_crons = _parse_yaml_crons(wf_path)
        if not yaml_crons:
            errores.append(f"{wf_name}: sin cron en el YAML (documentado como '{cron_doc}')")
            continue
        if len(yaml_crons) > 1:
            # Los workflows de update_* tienen 1 solo cron; si hay mas, revisar.
            errores.append(
                f"{wf_name}: {len(yaml_crons)} crons en YAML, runbook asume 1. "
                f"YAML={yaml_crons}, runbook='{cron_doc}'"
            )
            continue
        if yaml_crons[0] != cron_doc:
            errores.append(
                f"{wf_name}: drift. YAML='{yaml_crons[0]}' runbook='{cron_doc}'"
            )

    assert not errores, "Drift runbook<->workflow:\n  " + "\n  ".join(errores)


def test_runbook_cubre_los_update_workflows_existentes():
    """Los workflows update_* deben estar en el runbook (no basta con que
    los del runbook existan)."""
    runbook = _parse_runbook_crons()
    yaml_update = {p.name for p in WF_DIR.glob("update_*.yml")}
    faltan = yaml_update - set(runbook.keys())
    assert not faltan, (
        "workflows update_* sin documentar en el runbook: "
        + ", ".join(sorted(faltan))
    )
