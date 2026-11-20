# -*- coding: utf-8 -*-
"""Centinela: la tabla de workflows en 02_ARQUITECTURA coincide con los YAML.

Complementa test_runbook_vs_workflows.py: aquel verifica el runbook (procedimiento);
este verifica la arquitectura (tabla de inventario). Ambos protegen contra drift
entre documentacion y workflows reales.

Alcance: workflows con cron simple (5 campos POSIX). Multi-slot (daily_run con
5 disparos) se excluye explicitamente: su formato `17 23 / 17 3 / ... (5 slots)`
es documentado como tal y no se valida aqui.
"""
from __future__ import annotations

import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARQ = ROOT / "Consolidacion_Documentos" / "02_ARQUITECTURA.md"
WF_DIR = ROOT / ".github" / "workflows"

# Filas de tabla: | workflow.yml | cron | descripcion |
# Cron simple: 5 campos POSIX separados por espacios
ARQ_ROW_RE = re.compile(
    r"^\|\s*([a-z0-9_]+\.yml)\s*\|\s*([\d\*\/,\-\s]+?)\s*\|\s*.+?\s*\|[ \t]*$",
    re.MULTILINE,
)
CRON_SIMPLE_RE = re.compile(r"^[\d\*\/\,\-]+(?:\s+[\d\*\/\,\-]+){4}$")


def _parse_arquitectura_crons() -> dict[str, str]:
    """Devuelve {workflow: cron} para crons simples de 5 campos."""
    text = ARQ.read_text(encoding="utf-8-sig")
    out = {}
    for m in ARQ_ROW_RE.finditer(text):
        wf = m.group(1).strip()
        cron = m.group(2).strip()
        # Solo crons simples de 5 campos (excluye multi-slot como daily_run)
        if CRON_SIMPLE_RE.match(cron):
            out[wf] = cron
    return out


def _parse_yaml_crons(path: pathlib.Path) -> list[str]:
    """Devuelve la lista de crons del workflow (0..N)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        return []
    on = data.get("on")
    if on is None:
        on = data.get(True)  # PyYAML interpreta 'on:' como True
    if not isinstance(on, dict):
        return []
    schedule = on.get("schedule")
    if not isinstance(schedule, list):
        return []
    return [str(s.get("cron", "")).strip() for s in schedule
            if isinstance(s, dict) and s.get("cron")]


def test_arquitectura_crons_coinciden_con_yaml():
    """Cada cron simple en la tabla debe coincidir con el YAML real."""
    arq = _parse_arquitectura_crons()
    assert arq, "la tabla de arquitectura no expone crons simples"

    errores = []
    for wf_name, cron_doc in arq.items():
        wf_path = WF_DIR / wf_name
        if not wf_path.exists():
            errores.append(f"{wf_name}: en tabla pero no existe en .github/workflows/")
            continue
        yaml_crons = _parse_yaml_crons(wf_path)
        if not yaml_crons:
            errores.append(f"{wf_name}: sin cron en YAML (documentado como '{cron_doc}')")
            continue
        if len(yaml_crons) != 1:
            # Multi-slot documentado como tal (daily_run): no debe estar en la tabla
            errores.append(
                f"{wf_name}: {len(yaml_crons)} crons en YAML, tabla asume 1. "
                f"YAML={yaml_crons}, tabla='{cron_doc}'"
            )
            continue
        if yaml_crons[0] != cron_doc:
            errores.append(f"{wf_name}: drift. YAML='{yaml_crons[0]}' tabla='{cron_doc}'")

    assert not errores, "Drift arquitectura<->workflow:\n  " + "\n  ".join(errores)


def test_arquitectura_cubre_todos_los_crons_simples():
    """Todo YAML con cron simple (1 solo) debe estar en la tabla."""
    arq = _parse_arquitectura_crons()
    faltan = []
    for wf_path in WF_DIR.glob("*.yml"):
        yaml_crons = _parse_yaml_crons(wf_path)
        if len(yaml_crons) == 1:  # Solo crons simples (excluye multi-slot)
            wf_name = wf_path.name
            if wf_name not in arq:
                faltan.append(f"{wf_name}: cron '{yaml_crons[0]}' no documentado")
    assert not faltan, (
        "workflows con cron simple sin documentar en arquitectura:\n  "
        + "\n  ".join(sorted(faltan))
    )


def test_anti_zombie_cron_viejo_13f():
    """El cron trimestral viejo de 13F no debe reaparecer en la tabla."""
    arq = _parse_arquitectura_crons()
    assert arq.get("update_sec_13f.yml") != "17 6 20 2,5,8,11 *", (
        "cron trimestral viejo detectado en tabla de arquitectura. "
        "El ancla correcta es ventana+margen (mensual dia 10). "
        "Ver SEC-13F-PUBLICATION-ANCHOR."
    )
