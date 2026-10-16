#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera INDEX_H1_H18.md desde REGISTRO.yaml (Fase 5).

Vista derivada. Nunca se edita a mano. Uso:
    py scripts/genera_indice_auditorias.py --apply
    py scripts/genera_indice_auditorias.py         (dry-run)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTRO = ROOT / "docs" / "gobernanza" / "REGISTRO.yaml"
OUT = ROOT / "docs" / "auditoria" / "radar" / "audits" / "INDEX_H1_H18.md"

HEADER = """# Indice consolidado H1-H18 (auditoria externa del 2026-10-15)

**Generado automaticamente desde `docs/gobernanza/REGISTRO.yaml`.**
**NO editar a mano.** Regenerar con:

    py scripts/genera_indice_auditorias.py --apply

Puerta de entrada a los informes originales y a los tests de regresion
de cada hallazgo. Ver `docs/auditoria/radar/audits/` y los commits
referenciados.

| ID | Descripcion | Estado | Commit | Test | Razon |
|---|---|---|---|---|---|
"""


def _fmt(v):
    if v in (None, ""):
        return "-"
    return str(v).replace("|", "/")


def generar():
    data = yaml.safe_load(REGISTRO.read_text(encoding="utf-8"))
    items = [it for it in (data.get("items") or [])
             if it.get("tipo") == "auditoria_externa"
             and it.get("ciclo") == "2026-10-15"]

    def _key(it):
        iid = it.get("id", "")
        try:
            return (0, int(iid.lstrip("H")))
        except ValueError:
            return (1, iid)

    items.sort(key=_key)
    filas = []
    for it in items:
        filas.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} |".format(
                _fmt(it.get("id")),
                _fmt(it.get("descripcion")),
                _fmt(it.get("estado")),
                _fmt(it.get("commit")),
                _fmt(it.get("test")),
                _fmt(it.get("razon")),
            )
        )
    return HEADER + "\n" + "\n".join(filas) + "\n\n**Fin del indice.**\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="escribe el fichero (default: dry-run)")
    args = ap.parse_args()
    if not REGISTRO.exists():
        print("[ERROR] REGISTRO.yaml no existe")
        return 1
    nuevo = generar()
    if args.apply:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(nuevo, encoding="utf-8", newline="")
        print("[OK] escrito " + str(OUT))
        return 0
    actual = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
    if actual == nuevo:
        print("[OK] indice actualizado")
        return 0
    print("[DIFF] indice desactualizado. Ejecutar: "
          "py scripts/genera_indice_auditorias.py --apply")
    return 1


if __name__ == "__main__":
    sys.exit(main())
