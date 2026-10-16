# -*- coding: utf-8 -*-
"""Test selectivo de anclas temporales (Fase 6).

Detecta literales ISO en asignaciones a nivel modulo o como defaults
de funcion en codigo de produccion. Cada literal detectado debe estar
en ANCLAS.yaml.
"""
import ast
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ANCLAS_YAML = ROOT / "docs" / "gobernanza" / "ANCLAS.yaml"

EXCLUDE_FILES = {
    "calibrate_sow_wf.py",
    "calibrate_sow_wf_phase3.py",
    "calibrate_wyckoff_sow_5b3.py",
    "calibrate_wyckoff_sow_5b4.py",
    "calibrate_wyckoff_sow_5b4bis.py",
    "walk_forward_sow.py",
    "walk_forward_sow_anual.py",
    "validate_wyckoff_sow_5bX.py",
    "qa_precond_5bX.py",
    "compare_sow_candidates.py",
    "download_extended_history.py",
    "power_analysis_5bX.py",
}
EXCLUDE_PREFIX = "iae_"

ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _cargar():
    return yaml.safe_load(ANCLAS_YAML.read_text(encoding="utf-8"))


def _anclas_indexadas():
    """dict {(relpath, linea): (valor, ancla_id)}."""
    idx = {}
    for a in _cargar().get("anclas", []):
        for loc in a.get("ubicaciones", []):
            relpath, linea = loc.rsplit(":", 1)
            idx[(relpath, int(linea))] = (a["valor"], a["id"])
    return idx


def _es_excluido(rel):
    p = Path(rel)
    if p.parts[0] == "tests":
        return True
    if p.parts[0] == "scripts":
        if p.name in EXCLUDE_FILES:
            return True
        if p.name.startswith(EXCLUDE_PREFIX):
            return True
        if p.name.endswith(("_5bX.py", "_5b3.py", "_5b4.py", "_5b4bis.py")):
            return True
    return False


def _literales_iso(path):
    """[(linea, valor)] de literales ISO en asignaciones modulo o defaults."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    out = []
    # Asignaciones a nivel modulo
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and isinstance(node.value, ast.Constant):
                    v = node.value.value
                    if isinstance(v, str) and ISO_RE.match(v):
                        out.append((node.lineno, v))
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.value, ast.Constant):
                v = node.value.value
                if isinstance(v, str) and ISO_RE.match(v):
                    out.append((node.lineno, v))
    # Defaults de funcion
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.args.defaults + [x for x in node.args.kw_defaults if x]:
                if isinstance(d, ast.Constant):
                    v = d.value
                    if isinstance(v, str) and ISO_RE.match(v):
                        out.append((d.lineno, v))
    return out


def _escanea():
    hallazgos = []
    for base in ["src", "scripts"]:
        for path in (ROOT / base).rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            if _es_excluido(rel):
                continue
            for linea, valor in _literales_iso(path):
                hallazgos.append((rel, linea, valor))
    return hallazgos


def test_todas_las_anclas_documentadas():
    """Cada literal ISO detectado debe estar en ANCLAS.yaml."""
    idx = _anclas_indexadas()
    sin_doc = []
    for rel, linea, valor in _escanea():
        if (rel, linea) not in idx:
            sin_doc.append("{0}:{1} = {2!r}".format(rel, linea, valor))
    assert not sin_doc, (
        "Fechas ISO sin ancla documentada en ANCLAS.yaml:\n  "
        + "\n  ".join(sin_doc)
    )


def test_anclas_apuntan_a_sitios_existentes():
    """Cada ubicacion del ANCLAS.yaml debe existir."""
    faltantes = []
    for (rel, linea), (valor, aid) in _anclas_indexadas().items():
        path = ROOT / rel
        if not path.exists():
            faltantes.append("{0}: {1} no existe".format(aid, rel))
            continue
        lineas = path.read_text(encoding="utf-8").splitlines()
        if linea > len(lineas):
            faltantes.append("{0}: {1}:{2} fuera de rango".format(aid, rel, linea))
            continue
        if valor not in lineas[linea - 1]:
            faltantes.append(
                "{0}: {1}:{2} no contiene {3!r}".format(
                    aid, rel, linea, valor
                )
            )
    assert not faltantes, "\n".join(faltantes)


def test_categorias_validas():
    validas = {"serie_fuente", "frontera_contrato", "incidente_proveedor",
               "periodo_congelado", "calibracion", "otro"}
    for a in _cargar().get("anclas", []):
        assert a["categoria"] in validas, (
            "ancla {0}: categoria invalida {1!r}".format(a["id"], a["categoria"])
        )


def test_razon_minima():
    for a in _cargar().get("anclas", []):
        assert len(a.get("razon", "")) >= 20, (
            "ancla {0}: razon < 20 chars".format(a["id"])
        )


def test_ids_unicos():
    ids = [a["id"] for a in _cargar().get("anclas", [])]
    assert len(ids) == len(set(ids)), "ids de ancla duplicados"


def test_deteccion_detecta_literal_sin_ancla(tmp_path, monkeypatch):
    """El detector debe encontrar un literal ISO sin ancla documentada."""
    fake = tmp_path / "fake.py"
    fake.write_text("X = '2099-12-31'\n", encoding="utf-8")
    out = _literales_iso(fake)
    assert (1, "2099-12-31") in out