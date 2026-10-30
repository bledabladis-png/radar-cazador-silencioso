# -*- coding: utf-8 -*-
"""Contrato AST: render_mte solo lee campos que compute_mte retorna.

Clase de bug 1fb791ab (2026-10-30): render_mte anadio
mte_result.get('futures_status') pero el return de compute_mte no
exponia esa clave. Solo se escribia en el JSON del state. Test unit
con dict mockeado pasaba; integracion real nunca lo ejercito. Se
detecto en un run local 8h despues del fix.

Este test previene la clase entera:
1. Parsea render_mte (src/report/volatility_mte.py) y extrae todas
   las claves que lee via mte_result.get('X') o mte_result['X'].
2. Parsea compute_mte (indicators/mte/engine.py) y extrae las claves
   del return { ... }.
3. Falla si alguna clave leida por render_mte no esta en el return.

Contrato: cualquier mte_result.get('X') nuevo en render_mte debe
venir acompanado de 'X' en el return de compute_mte. Si el campo vive
solo en el state JSON, no es accesible desde el render.
"""
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE = REPO_ROOT / "indicators" / "mte" / "engine.py"
RENDER = REPO_ROOT / "src" / "report" / "volatility_mte.py"

# Campos del state JSON que viven SOLO en el state por diseno. NO
# deben estar en el return porque su consumidor natural no es el
# reporte diario: schema_version y temporal_contract_version son
# metadatos del schema; effective_date / expected_date / coverage ya
# se publican en el bloque Provenance y en la tabla "Alineacion con
# sesion esperada". Whitelist cerrada: si crece, actualizar
# test_whitelist_state_only_justificada con justificacion.
STATE_ONLY_WHITELIST = frozenset({
    "schema_version",
    "temporal_contract_version",
    "effective_date",
    "expected_date",
    "coverage",
})


def _find_function(tree, name):
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name):
            return node
    return None


def _compute_mte_return_keys():
    """Claves del return { ... } de compute_mte en engine.py."""
    tree = ast.parse(ENGINE.read_text(encoding="utf-8-sig"))
    fn = _find_function(tree, "compute_mte")
    assert fn is not None, "compute_mte no encontrado en engine.py"
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys = set()
            for k in node.value.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
            return keys
    raise AssertionError("return { ... } no encontrado en compute_mte")


def _render_mte_read_keys():
    """Claves que render_mte lee via mte_result.get('X') o mte_result['X']."""
    tree = ast.parse(RENDER.read_text(encoding="utf-8-sig"))
    fn = _find_function(tree, "render_mte")
    assert fn is not None, "render_mte no encontrado en volatility_mte.py"
    keys = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "mte_result"):
                if (node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)):
                    keys.add(node.args[0].value)
        if isinstance(node, ast.Subscript):
            if (isinstance(node.value, ast.Name)
                    and node.value.id == "mte_result"
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                keys.add(node.slice.value)
    return keys


def test_render_mte_no_lee_campos_ausentes_en_return():
    """Cualquier clave leida por render_mte debe estar en el return.

    Si falla, el mensaje indica exactamente que clave anadir al
    return de compute_mte en engine.py.
    """
    returns = _compute_mte_return_keys()
    reads = _render_mte_read_keys()

    assert reads, (
        "render_mte no lee ninguna clave de mte_result. "
        "El AST no encontro el patron esperado (mte_result.get o "
        "mte_result[...]). Revisar si el nombre del parametro cambio."
    )

    missing = reads - returns
    assert not missing, (
        f"render_mte lee {sorted(missing)} pero compute_mte no los "
        f"retorna. Clase de bug 1fb791ab: el campo vive en el state "
        f"JSON pero no en el return, y render_mte no lo puede pintar. "
        f"Anadir al return de compute_mte (engine.py). "
        f"Return actual: {sorted(returns)}"
    )


def test_render_mte_lee_futures_status_y_pending():
    """Regresion explicita: los 2 campos que ya fallaron estan cubiertos.

    Cualquier refactor de render_mte que elimine estas lecturas debe
    ser consciente. Si se eliminan, este test falla y obliga a
    revisar el contrato.
    """
    reads = _render_mte_read_keys()
    assert "futures_status" in reads, (
        "render_mte dejo de leer futures_status. El reporte perderia "
        "la advertencia de inputs de futuros stale. Si es intencional, "
        "borrar este test con justificacion."
    )
    assert "pending" in reads, (
        "render_mte dejo de leer pending. El reporte perderia el "
        "aviso de transicion de escenario. Si es intencional, borrar "
        "este test con justificacion."
    )


def _compute_mte_state_keys():
    """Claves del dict de json.dump({...}) dentro de compute_mte.

    Parsea el bloque de persistencia del state JSON. Es la fuente
    de verdad de lo que se publica en outputs/state/mte_state.json.
    """
    tree = ast.parse(ENGINE.read_text(encoding="utf-8-sig"))
    fn = _find_function(tree, "compute_mte")
    assert fn is not None, "compute_mte no encontrado en engine.py"
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            # json.dump({...}, f, ...)
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr == "dump"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "json"):
                if node.args and isinstance(node.args[0], ast.Dict):
                    keys = set()
                    for k in node.args[0].keys:
                        if (isinstance(k, ast.Constant)
                                and isinstance(k.value, str)):
                            keys.add(k.value)
                    return keys
    raise AssertionError("json.dump({...}) no encontrado en compute_mte")


def test_compute_mte_state_no_omite_campos_en_return():
    """Cualquier campo del state JSON (excepto whitelist) debe estar
    tambien en el return.

    Cierra la clase 1fb791ab en direccion upstream. El bug original:
    futures_status vivia en el state JSON, no en el return. El test
    downstream (render subconjunto return) solo lo detecta si render
    intenta leerlo. Si render nunca lo intenta, el campo queda oculto
    indefinidamente: el operador lo ve en el JSON, pero el reporte
    no. Este test detecta la omision en el origen.

    Caso concreto hoy: state publica schema_version, temporal_
    contract_version, effective_date, expected_date, coverage,
    futures_status, scenario, confidence, msi, ipi, srs, shs, cls,
    ips, pending. Los 5 primeros son whitelist (solo state). Los
    otros 10 deben estar en el return.
    """
    state_keys = _compute_mte_state_keys()
    return_keys = _compute_mte_return_keys()

    missing = (state_keys - STATE_ONLY_WHITELIST) - return_keys
    assert not missing, (
        f"El state JSON de compute_mte publica {sorted(missing)} pero "
        f"el return no los incluye. Clase 1fb791ab: un campo que solo "
        f"vive en el state es invisible desde el render, aunque el "
        f"operador pueda leerlo del JSON. Anadir al return de "
        f"compute_mte (engine.py). "
        f"State actual: {sorted(state_keys)}. "
        f"Return actual: {sorted(return_keys)}. "
        f"Whitelist (solo state, por diseno): "
        f"{sorted(STATE_ONLY_WHITELIST)}."
    )


def test_whitelist_state_only_justificada():
    """La whitelist de campos state-only no crece sin justificacion.

    Si alguien anade un campo a la whitelist sin justificar, este
    test falla y obliga a actualizarlo con la nueva lista + razon
    en el mensaje del commit. Previene "whitelist creep": que un
    campo funcional se escape del return simplemente ignorandolo.
    """
    assert STATE_ONLY_WHITELIST == frozenset({
        "schema_version",
        "temporal_contract_version",
        "effective_date",
        "expected_date",
        "coverage",
    }), (
        "La whitelist de campos state-only cambio. Si es una "
        "decision consciente, actualizar este test con la nueva "
        "lista y justificar en el mensaje del commit. Si no, "
        "revisar: puede ser un campo funcional que deberia estar "
        "en el return."
    )


def test_state_y_return_comparten_campos_funcionales():
    """Regresion explicita: futures_status y pending estan en ambos.

    Los 2 campos que ya fallaron una vez (futures_status en 5637d45e,
    pending en 5b96192f) deben estar en state Y return. Si un refactor
    los saca del return pero los deja en state, este test lo detecta
    inmediatamente.
    """
    state_keys = _compute_mte_state_keys()
    return_keys = _compute_mte_return_keys()
    for campo in ("futures_status", "pending"):
        assert campo in state_keys, (
            f"{campo} desaparecio del state JSON. Regresion: el "
            f"reporte no podria pintarlo aunque el operador lo viera "
            f"en el JSON."
        )
        assert campo in return_keys, (
            f"{campo} desaparecio del return. Regresion de la clase "
            f"1fb791ab: render_mte no podria pintarlo."
        )
