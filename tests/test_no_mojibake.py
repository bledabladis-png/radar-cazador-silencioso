# -*- coding: utf-8 -*-
"""Centinela: prohibe mojibake UTF-8->Latin1 en strings de codigo productivo.

Detecta la clase: string guardado como UTF-8 dos veces (e.g. "seÃ±ales"
en vez de "señales"). Bug observado en src/report/header.py (sesion 23).

Patron: Ã*, Â*, â€*. Si aparece un mojibake nuevo no catalogado, ampliar
PARES en el helper de deteccion.
"""

import ast
import pathlib

PARES_MOJIBAKE = [
    "Ã", "Â", "â€",
]

DIRS_PRODUCTIVOS = ["src", "regimes", "indicators", "scripts", "config"]
EXCLUIR_PARTS = {".git", "__pycache__", "venv", ".venv", "node_modules", "outputs", "data", "tests"}


def tiene_mojibake(s: str) -> bool:
    return any(pat in s for pat in PARES_MOJIBAKE)


def _strings_de_fichero(py: pathlib.Path):
    try:
        tree = ast.parse(py.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, SyntaxError):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value
        if isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    yield node.lineno, part.value


def test_no_mojibake_en_strings_productivos():
    raiz = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for d in DIRS_PRODUCTIVOS:
        dp = raiz / d
        if not dp.exists():
            continue
        for py in dp.rglob("*.py"):
            if any(p in EXCLUIR_PARTS for p in py.parts):
                continue
            for linea, s in _strings_de_fichero(py):
                if tiene_mojibake(s):
                    offenders.append(f"{py.relative_to(raiz)}:{linea}: {s[:80]!r}")
    assert not offenders, (
        "Mojibake detectado en strings productivos (UTF-8 doble-codificado). "
        "Fix: sustituir por caracter correcto. " + str(offenders)
    )
