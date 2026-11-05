# -*- coding: utf-8 -*-
"""Centinela repo-wide: detecta mojibake UTF-8->Latin-1->UTF-8 por bytes.

Complementa test_no_mojibake.py, que usa AST y solo mira strings
Python productivos. Este escanea cualquier fichero de texto del repo.

Deteccion heuristica por rangos Unicode (no lista fija de patrones):
  - 2-byte UTF-8 doble-codificado: U+00C0-U+00FF seguido de
    U+0080-U+00BF. Cubre: e-acute, section, middot, n-tilde, etc.
  - 3-byte UTF-8 doble-codificado: U+00E0-U+00EF + (U+2018-U+203A
    o U+0080-U+00BF) + U+0080-U+00BF. Cubre: euro-duplicado,
    comillas-tipograficas, em-dash, etc.

Clase cp437-sobre-utf-8 (box-drawing + Latin-1) fuera de alcance:
cubierta por test_no_mojibake.py via PARES_MOJIBAKE para strings
Python productivos. En docs, si aparece, se anade aqui.

4-byte UTF-8 (emoji doble-codificado) fuera de alcance: probabilidad
baja, deteccion compleja por longitud de secuencia.

Bug real: REPORT-ENCODING-MOJIBAKE-BITACORA (S29, 05_BITACORA L2 y L1777).

Roots: auto-descubrimiento. Ficheros en raiz + recursivo en todos los
directorios top-level excepto EXCLUDE_DIRS. Directorio nuevo entra
solo. Si un dir debe salir, se anade a EXCLUDE_DIRS con justificacion.

Whitelist: ficheros que documentan el bug con ejemplos historicos.
Por fichero, no por (fichero, patron, conteo): los conteos no envejecen.
"""
from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parent.parent

EXCLUDE_DIRS = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "outputs",
    "data",
    "__pycache__",
    "venv",
    ".venv",
    "node_modules",
    "dist",
    "build",
    ".idea",
    ".vscode",
}
# Justificacion:
#  - outputs/, data/: artefactos de proveedores con calendarios no-NYSE
#    (ver 00_ARRANQUE 2, segundo candado temporal). No controlamos
#    su encoding.
#  - .git, __pycache__, caches, venv, node_modules, dist, build,
#    .idea, .vscode: infraestructura, binarios o dependencias externas.

EXTS = {".py", ".md", ".yaml", ".yml", ".csv", ".json", ".txt"}

WHITELIST_FILES = {
    "docs/auditoria/s03_deep/report_iae_contexts.md",
    "Consolidacion_Documentos/04_HISTORICO.md",
}


def _es_2byte_utf8_duplicado(o1, o2):
    return 0xC0 <= o1 <= 0xFF and 0x80 <= o2 <= 0xBF


def _es_3byte_utf8_duplicado(o1, o2, o3):
    if not (0xE0 <= o1 <= 0xEF):
        return False
    seg2_ok = (0x2018 <= o2 <= 0x203A) or (0x80 <= o2 <= 0xBF)
    seg3_ok = 0x80 <= o3 <= 0xBF
    return seg2_ok and seg3_ok


def _scan_text(texto):
    """Devuelve lista de (kind, pos, segmento) para hits en texto."""
    hits = []
    n = len(texto)
    for i in range(n - 1):
        o1 = ord(texto[i])
        o2 = ord(texto[i + 1])
        if _es_2byte_utf8_duplicado(o1, o2):
            hits.append(("2byte", i, texto[i:i + 2]))
            continue
        if i + 2 < n:
            o3 = ord(texto[i + 2])
            if _es_3byte_utf8_duplicado(o1, o2, o3):
                hits.append(("3byte", i, texto[i:i + 3]))
    return hits


def _iter_candidatos(raiz):
    """Itera ficheros a escanear: raiz + subdirs no excluidos."""
    for f in raiz.iterdir():
        if f.is_file():
            yield f
        elif f.is_dir() and f.name not in EXCLUDE_DIRS:
            yield from f.rglob("*")


def _scan_bytes(raiz=None):
    """Devuelve dict {(rel_path, kind): [(pos, seg), ...]}."""
    raiz = pathlib.Path(raiz) if raiz is not None else ROOT
    found = {}
    for f in _iter_candidatos(raiz):
        if not f.is_file() or f.suffix not in EXTS:
            continue
        try:
            texto = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits = _scan_text(texto)
        if not hits:
            continue
        rel = f.relative_to(raiz).as_posix()
        for kind, pos, seg in hits:
            found.setdefault((rel, kind), []).append((pos, seg))
    return found


def test_no_mojibake_en_ficheros():
    found = _scan_bytes()
    unexpected = {
        k: v for k, v in found.items()
        if k[0] not in WHITELIST_FILES
    }
    assert not unexpected, (
        "Mojibake UTF-8 doble-codificado fuera de whitelist. "
        "Limpiar con bytes.replace o anadir el fichero a "
        "WHITELIST_FILES con justificacion (evidencia historica). "
        "Encontrado: " + str({
            f: len(v) for (f, _kind), v in unexpected.items()
        })
    )


def test_auto_descubre_dir_nuevo(tmp_path):
    """Centinela: dir top-level nuevo con mojibake debe detectarse."""
    nuevo = tmp_path / "nuevo_dir_sintetico"
    nuevo.mkdir()
    contenido = "R" + chr(0xC3) + chr(0xA9) + "gimen"
    (nuevo / "ejemplo.md").write_text(contenido, encoding="utf-8")
    found = _scan_bytes(raiz=tmp_path)
    keys = list(found.keys())
    assert any(k[0] == "nuevo_dir_sintetico/ejemplo.md" for k in keys), (
        "Auto-descubrimiento no detecto mojibake en dir nuevo. "
        "Encontrado: " + str(keys)
    )


def test_auto_descubre_fichero_raiz(tmp_path):
    """Centinela: ficheros en raiz (no en subdir) tambien se escanean."""
    contenido = "R" + chr(0xC3) + chr(0xA9) + "gimen"
    (tmp_path / "raiz.md").write_text(contenido, encoding="utf-8")
    found = _scan_bytes(raiz=tmp_path)
    keys = list(found.keys())
    assert any(k[0] == "raiz.md" for k in keys), (
        "Fichero en raiz no detectado. Encontrado: " + str(keys)
    )


def test_excluye_outputs_y_data(tmp_path):
    """Centinela: outputs/ y data/ no se escanean (artefactos externos)."""
    contenido = "R" + chr(0xC3) + chr(0xA9) + "gimen"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "sintetico.md").write_text(contenido, encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    (data / "sintetico.csv").write_text(contenido, encoding="utf-8")
    found = _scan_bytes(raiz=tmp_path)
    assert not any("outputs/" in k[0] for k in found), (
        "outputs/ deberia estar excluido. Encontrado: " + str(list(found))
    )
    assert not any("data/" in k[0] for k in found), (
        "data/ deberia estar excluido. Encontrado: " + str(list(found))
    )
