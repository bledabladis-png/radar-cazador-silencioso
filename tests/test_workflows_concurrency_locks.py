# -*- coding: utf-8 -*-
"""D9 (2026-09-30): invariante de concurrency.group por recurso.

Dos workflows que hacen `git add` del mismo path deben compartir el
mismo concurrency.group. Sin esto, pueden coincidir en vuelo y el
segundo push falla por conflicto de rebase (o sobreescribe cambios
del primero si el CSV se reescribe entero).

Caso origen: update_index_holdings.yml y update_european_holdings.yml
escribian ambos data/index_holdings.csv con grupos distintos.

Limitacion: comparacion exacta de paths. Globs (git add path/*) y
directorios con trailing slash no se expanden. Documentar y ampliar
si aparece un caso real.
"""
import re
from pathlib import Path

WORKFLOWS_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _parse_workflow(path):
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^\s*group:\s*(\S+)\s*$", text, re.MULTILINE)
    group = m.group(1) if m else None
    # git add <path> (stripped). Excluye lineas con `||` o pipes.
    adds = set()
    for line in text.splitlines():
        m2 = re.match(r"^\s*git add\s+([^\s|&;]+)", line)
        if m2:
            adds.add(m2.group(1))
    return group, adds


def test_workflows_que_escriben_mismo_path_comparten_group():
    workflows = sorted(WORKFLOWS_DIR.glob("*.yml"))
    assert workflows, "no se encontraron workflows en {}".format(WORKFLOWS_DIR)

    by_path = {}
    for wf in workflows:
        group, adds = _parse_workflow(wf)
        for path in adds:
            by_path.setdefault(path, []).append((wf.name, group))

    conflicts = []
    for path, entries in by_path.items():
        groups = {g for _, g in entries if g}
        if len(groups) > 1:
            conflicts.append((path, entries))

    msg = "Workflows comparten path sin mismo concurrency.group:\n"
    for path, entries in conflicts:
        msg += "  {}: {}\n".format(path, entries)
    assert not conflicts, msg


def test_index_holdings_csv_comparte_group():
    """Regresion del caso D9: los dos workflows que escriben
    data/index_holdings.csv deben usar update_index_holdings_csv."""
    a = WORKFLOWS_DIR / "update_index_holdings.yml"
    b = WORKFLOWS_DIR / "update_european_holdings.yml"
    ga, _ = _parse_workflow(a)
    gb, _ = _parse_workflow(b)
    assert ga == gb, "grupos distintos: {} vs {}".format(ga, gb)
    assert ga == "update_index_holdings_csv", "grupo inesperado: {}".format(ga)
