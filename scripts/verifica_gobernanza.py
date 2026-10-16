#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verificador de gobernanza. Exit 0 si OK, exit 1 con errores."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTRO = ROOT / "docs" / "gobernanza" / "REGISTRO.yaml"
HANDOFF = ROOT / "docs" / "handoffs" / "HANDOFF.md"

VALID_ESTADOS = frozenset({
    "OPEN", "IN_PROGRESS", "BLOCKED_EXTERNAL",
    "FIXED", "WONT_FIX", "FALSE_POSITIVE", "SUPERSEDED",
})
REQUIRE_RAZON = frozenset({"WONT_FIX", "FALSE_POSITIVE", "SUPERSEDED"})
RAZON_MIN = 20
EVIDENCIA_MIN = 20
RAZONES_GENERICAS = frozenset({
    "wont fix", "no aplica", "n/a", "sin razon", "descartado",
})


def _git_commit_exists(sha):
    if not sha or sha == "PENDING":
        return None
    try:
        r = subprocess.run(
            ["git", "cat-file", "-e", sha],
            cwd=str(ROOT), capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _collect_handoff_pendientes(handoff_path):
    """Lineas de la seccion 4 del HANDOFF que empiezan por "- ".

    handoff_path es obligatorio para evitar dependencia de variable
    global (mala testabilidad).
    """
    if not handoff_path or not handoff_path.exists():
        return []
    text = handoff_path.read_text(encoding="utf-8")
    i = text.find("## 4. PENDIENTES VIVOS")
    if i < 0:
        return []
    j = text.find("## 5.", i)
    if j < 0:
        j = len(text)
    bloque = text[i:j]
    return [ln.strip() for ln in bloque.splitlines()
            if ln.strip().startswith("- ")]


def verificar(registro_path=None, handoff_path=None):
    """Verifica gobernanza.

    Parametros opcionales para tests. Default: paths de modulo.
    """
    if registro_path is None:
        registro_path = REGISTRO
    if handoff_path is None:
        handoff_path = HANDOFF

    errores = []
    if not registro_path.exists():
        return [f"REGISTRO.yaml no existe: {registro_path}"]
    try:
        data = yaml.safe_load(registro_path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"YAML invalido: {e}"]
    if not isinstance(data, dict) or data.get("version") != 1:
        errores.append("version != 1")
    items = data.get("items") or []
    if not isinstance(items, list):
        errores.append("items no es lista")
        return errores

    ids_vistos = set()
    handoff_pendientes = _collect_handoff_pendientes(handoff_path)

    for it in items:
        iid = it.get("id", "<sin id>")
        if iid in ids_vistos:
            errores.append(f"[{iid}] ID duplicado")
        ids_vistos.add(iid)

        estado = it.get("estado")
        if estado not in VALID_ESTADOS:
            errores.append(f"[{iid}] estado invalido: {estado!r}")

        if estado == "FIXED":
            commit = it.get("commit")
            test = it.get("test")
            evid = it.get("evidencia")
            if not commit:
                errores.append(f"[{iid}] FIXED sin commit")
            elif commit != "PENDING":
                if _git_commit_exists(commit) is False:
                    errores.append(f"[{iid}] commit no existe: {commit}")
            has_test = bool(test)
            has_evid = isinstance(evid, str) and len(evid) >= EVIDENCIA_MIN
            if not has_test and not has_evid:
                errores.append(
                    f"[{iid}] FIXED sin test ni evidencia >= {EVIDENCIA_MIN}"
                )
            if has_test and commit != "PENDING":
                if not (ROOT / test).exists():
                    errores.append(f"[{iid}] test no existe: {test}")

        if estado in REQUIRE_RAZON:
            razon = it.get("razon") or ""
            razon_l = razon.strip().lower()
            if razon_l in RAZONES_GENERICAS:
                errores.append(f"[{iid}] razon generica: {razon!r}")
            elif len(razon) < RAZON_MIN:
                errores.append(
                    f"[{iid}] razon < {RAZON_MIN} chars para {estado}"
                )

        if estado == "BLOCKED_EXTERNAL" and not it.get("dependencia"):
            errores.append(f"[{iid}] BLOCKED_EXTERNAL sin dependencia")

        if estado in {"OPEN", "IN_PROGRESS"} and not it.get("siguiente_accion"):
            errores.append(f"[{iid}] {estado} sin siguiente_accion")

        if estado != "FIXED":
            if any(iid in ln for ln in handoff_pendientes):
                errores.append(
                    f"[{iid}] estado={estado} aparece como pendiente en HANDOFF"
                )

    return errores


def main():
    errores = verificar()
    if not errores:
        print("[OK] Gobernanza coherente")
        return 0
    print(f"[FAIL] {len(errores)} errores de gobernanza:")
    for e in errores:
        print(f"  - {e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())