#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verificador de gobernanza. Exit 0 si OK, exit 1 con errores."""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTRO = ROOT / "docs" / "gobernanza" / "REGISTRO.yaml"
HANDOFF = ROOT / "HANDOFF - Radar de Rotacion Sectorial.txt"

VALID_ESTADOS = frozenset({
    "OPEN", "IN_PROGRESS", "BLOCKED_EXTERNAL",
    "FIXED", "WONT_FIX", "FALSE_POSITIVE", "SUPERSEDED",
})
REQUIRE_RAZON = frozenset({"WONT_FIX", "FALSE_POSITIVE", "SUPERSEDED"})
ESTADOS_CERRADOS = frozenset({
    "FIXED", "WONT_FIX", "FALSE_POSITIVE", "SUPERSEDED",
})
RAZON_MIN = 20
EVIDENCIA_MIN = 20
RAZONES_GENERICAS = frozenset({
    "wont fix", "no aplica", "n/a", "sin razon", "descartado",
})

# Delimitadores que definen un ID como token completo. Evita que "H5"
# matchee dentro de "H5.3" o "H5.3b". Un ID debe estar precedido y
# seguido por un caracter que NO sea alfanumerico ni punto ni guion.
_ID_TOKEN_FMT = r"(?<![A-Za-z0-9_.\-]){0}(?![A-Za-z0-9_.\-])"


def _menciona_id(iid, lineas):
    """True si iid aparece como token completo en alguna linea."""
    pat = re.compile(_ID_TOKEN_FMT.format(re.escape(iid)))
    for ln in lineas:
        if pat.search(ln):
            return True
    return False


def _git_commit_exists(sha):
    """True si el commit existe Y es alcanzable desde alguna ref.

    2026-10-22 (REP-AUDIT-20261022-GOVERNANCE-SHA-REACHABILITY):
    `git cat-file -e` acepta objetos huerfanos (post-rebase, aun no
    recolectados). Un item FIXED con SHA huerfano apunta a un commit
    que desaparecera al correr git GC. Verificamos tambien
    reachability desde refs/heads/ o refs/tags/. Si for-each-ref
    falla, fallback conservador a True (no bloquear por error del
    subcomando).
    """
    if not sha or sha == "PENDING":
        return None
    try:
        r = subprocess.run(
            ["git", "cat-file", "-e", sha],
            cwd=str(ROOT), capture_output=True, timeout=10,
        )
        if r.returncode != 0:
            return False
        r2 = subprocess.run(
            ["git", "for-each-ref", "--contains", sha,
             "refs/heads/", "refs/tags/"],
            cwd=str(ROOT), capture_output=True, timeout=10,
        )
        if r2.returncode != 0:
            return True
        return bool(r2.stdout.strip())
    except Exception:
        return False


def _collect_handoff_pendientes(handoff_path):
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
    if registro_path is None:
        registro_path = REGISTRO
    if handoff_path is None:
        handoff_path = HANDOFF

    errores = []
    if not registro_path.exists():
        return ["REGISTRO.yaml no existe: {0}".format(registro_path)]
    try:
        data = yaml.safe_load(registro_path.read_text(encoding="utf-8"))
    except Exception as e:
        return ["YAML invalido: {0}".format(e)]
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
            errores.append("[{0}] ID duplicado".format(iid))
        ids_vistos.add(iid)

        estado = it.get("estado")
        if estado not in VALID_ESTADOS:
            errores.append("[{0}] estado invalido: {1!r}".format(iid, estado))

        if estado == "FIXED":
            commit = it.get("commit")
            test = it.get("test")
            evid = it.get("evidencia")
            if not commit:
                errores.append("[{0}] FIXED sin commit".format(iid))
            elif not isinstance(commit, str):
                # Bug 2026-10-28: YAML convierte SHAs solo-digitos
                # (p.ej. 64866361) a int. _git_commit_exists(int)
                # hace subprocess con int y cae en TypeError
                # capturado por except Exception, devolviendo False.
                # El item se reportaba erroneamente como "commit no
                # existe o es huerfano". El SHA debe ir entrecomillado.
                errores.append(
                    "[{0}] commit no es string (SHA entrecomillado "
                    "en YAML): {1!r} ({2})".format(
                        iid, commit, type(commit).__name__
                    )
                )
            elif commit != "PENDING":
                if _git_commit_exists(commit) is False:
                    errores.append(
                        "[{0}] commit no existe o es huerfano "
                        "(no alcanzable desde refs): {1}. Tras "
                        "rebase, actualizar al SHA actual en main "
                        "o reclasificar a WONT_FIX.".format(iid, commit)
                    )
            has_test = bool(test)
            has_evid = isinstance(evid, str) and len(evid) >= EVIDENCIA_MIN
            if not has_test and not has_evid:
                errores.append(
                    "[{0}] FIXED sin test ni evidencia >= {1}".format(
                        iid, EVIDENCIA_MIN
                    )
                )
            if has_test and commit != "PENDING":
                if not (ROOT / test).exists():
                    errores.append(
                        "[{0}] test no existe: {1}".format(iid, test)
                    )

        if estado in REQUIRE_RAZON:
            razon = it.get("razon") or ""
            razon_l = razon.strip().lower()
            if razon_l in RAZONES_GENERICAS:
                errores.append(
                    "[{0}] razon generica: {1!r}".format(iid, razon)
                )
            elif len(razon) < RAZON_MIN:
                errores.append(
                    "[{0}] razon < {1} chars para {2}".format(
                        iid, RAZON_MIN, estado
                    )
                )

        if estado == "BLOCKED_EXTERNAL" and not it.get("dependencia"):
            errores.append("[{0}] BLOCKED_EXTERNAL sin dependencia".format(iid))

        if estado in {"OPEN", "IN_PROGRESS"} and not it.get("siguiente_accion"):
            errores.append(
                "[{0}] {1} sin siguiente_accion".format(iid, estado)
            )

        # Regla 10: SOLO items cerrados no pueden resucitar en HANDOFF.
        # Items OPEN/IN_PROGRESS/BLOCKED_EXTERNAL son frentes vivos y
        # pueden (deben) aparecer como pendientes.
        if estado in ESTADOS_CERRADOS:
            if _menciona_id(iid, handoff_pendientes):
                errores.append(
                    "[{0}] estado={1} aparece como pendiente en HANDOFF "
                    "(reabre un item cerrado)".format(iid, estado)
                )

    # Check 2026-10-28: detectar "- id:" embebidos dentro de un
    # valor de string. Bug real: en b2f3e30f, el append de items al
    # final del REGISTRO no dejo linea en blanco. El "- id:
    # AUTONOMY..." quedo pegado al final de un siguiente_accion: |,
    # y las claves siguientes sobreescribieron el mapping del
    # anterior (PyYAML conserva el ultimo valor). GOVERNANZA-
    # HANDOFF-OBSOLETO quedo corrupto y AUTONOMY desaparecio.
    _texto = registro_path.read_text(encoding="utf-8")
    for _ln_no, _ln in enumerate(_texto.splitlines(), start=1):
        if "- id:" in _ln and not _ln.lstrip().startswith("- id:"):
            errores.append(
                "linea {0}: '- id:' embebido (no al inicio de "
                "linea, probable concatenacion sin separador): "
                "{1!r}".format(_ln_no, _ln.strip()[:100])
            )

    return errores


def main():
    errores = verificar()
    if not errores:
        print("[OK] Gobernanza coherente")
        return 0
    print("[FAIL] {0} errores de gobernanza:".format(len(errores)))
    for e in errores:
        print("  - {0}".format(e))
    return 1


if __name__ == "__main__":
    sys.exit(main())
