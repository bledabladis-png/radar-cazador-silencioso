# -*- coding: utf-8 -*-
"""Centinela repo-wide por bytes: detecta mojibake en cualquier
fichero de texto (docs, csv, yaml, json, md, py...).

Complementa test_no_mojibake.py, que usa AST y solo mira strings
Python productivos. Este escanea bytes para detectar reencoding
UTF-8 -> Latin-1 -> UTF-8 en docs y artefactos.

Bug real: REPORT-ENCODING-MOJIBAKE-BITACORA (S29, 05_BITACORA
L2 y L1777). Fix aplicado en 02_ARQUITECTURA.md L474.

Whitelist por (fichero, patron) -> conteo exacto. Evidencias
que documentan el bug con ejemplos historicos.
"""
from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parent.parent

ROOTS = (
    "src",
    "scripts",
    "tests",
    "docs",
    "Consolidacion_Documentos",
    ".github",
)

EXTS = {".py", ".md", ".yaml", ".yml", ".csv", ".json", ".txt"}

PATTERNS = {
    "doble-utf8 section":   b"\xc3\x82\xc2\xa7",
    "doble-utf8 middot":    b"\xc3\x82\xc2\xb7",
    "doble-utf8 a-acute":   b"\xc3\x83\xc2\xa1",
    "doble-utf8 e-acute":   b"\xc3\x83\xc2\xa9",
    "doble-utf8 i-acute":   b"\xc3\x83\xc2\xad",
    "doble-utf8 o-acute":   b"\xc3\x83\xc2\xb3",
    "doble-utf8 u-acute":   b"\xc3\x83\xc2\xba",
    "doble-utf8 n-tilde":   b"\xc3\x83\xc2\xb1",
    "doble-utf8 apostrofo": b"\xc3\xa2\xe2\x82\xac\xe2\x84\xa2",
    "doble-utf8 comilla-i": b"\xc3\xa2\xe2\x82\xac\xc5\x93",
    "doble-utf8 comilla-c": b"\xc3\xa2\xe2\x82\xac\xc2\x9d",
    "doble-utf8 en-dash":   b"\xc3\xa2\xe2\x82\xac\xe2\x80\x93",
    "doble-utf8 em-dash":   b"\xc3\xa2\xe2\x82\xac\xe2\x80\x9d",
    "cp437 u252c-u00ba":    b"\xe2\x94\xac\xc2\xba",
    "cp437 u252c-u00a7":    b"\xe2\x94\xac\xc2\xa7",
}

# Evidencias historicas que documentan el bug con ejemplos.
WHITELIST = {
    ("docs/auditoria/s03_deep/report_iae_contexts.md",
     "doble-utf8 e-acute"): 2,
    ("docs/auditoria/s03_deep/report_iae_contexts.md",
     "doble-utf8 o-acute"): 3,
    ("Consolidacion_Documentos/04_HISTORICO.md",
     "doble-utf8 e-acute"): 1,
    ("Consolidacion_Documentos/04_HISTORICO.md",
     "doble-utf8 o-acute"): 1,
}


def _scan_bytes():
    found = {}
    for r in ROOTS:
        p = ROOT / r
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if not f.is_file() or f.suffix not in EXTS:
                continue
            try:
                data = f.read_bytes()
            except OSError:
                continue
            rel = f.relative_to(ROOT).as_posix()
            for label, pat in PATTERNS.items():
                n = data.count(pat)
                if n:
                    found[(rel, label)] = n
    return found


def test_no_mojibake_en_ficheros():
    found = _scan_bytes()
    unexpected = {k: v for k, v in found.items() if k not in WHITELIST}
    assert not unexpected, (
        "Mojibake fuera de whitelist (bytes). "
        "Limpiar con bytes.replace o anadir a WHITELIST "
        "con justificacion. Encontrado: " + str(unexpected)
    )


def test_whitelist_no_stale():
    found = _scan_bytes()
    stale = []
    for (f, label), n in WHITELIST.items():
        got = found.get((f, label), 0)
        if got != n:
            stale.append((f, label, n, got))
    assert not stale, (
        "Whitelist stale (conteo cambio). Actualizar: " + str(stale)
    )
