# -*- coding: utf-8 -*-
"""Helper AST para contratos dict entre modulos del pipeline.

Clase de bug 1fb791ab (2026-10-30): un modulo A produce un dict (return
o state JSON), un modulo B lo consume via `A_output.get(X)` o
`A_output['X']`. Si A no publica X, B obtiene None silenciosamente
(con .get()) o KeyError (con [...]). Test unitario con dict mockeado
pasa; integracion real no ejercita la produccion.

Este helper implementa el contrato en 2 direcciones:

- Downstream (`assert_render_subset_return`):
  consumer lee X => producer return incluye X.
  Detecta "render pide campo que el productor no da".

- Upstream (`assert_state_subset_return`):
  producer state publica X (no whitelist) => producer return incluye X.
  Detecta "campo funcional que solo vive en el state JSON, invisible
  al render".

Ambas direcciones + whitelist justificada + test de regresion
explicita constituyen el contrato completo. Aplicado a MTE en
tests/test_mte_return_contract.py (2026-10-30).

Uso tipico:

    from tests._dict_contract import (
        producer_return_keys,
        producer_state_keys,
        consumer_read_keys,
        assert_render_subset_return,
        assert_state_subset_return,
    )

    def test_mi_contrato():
        assert_render_subset_return(
            producer_module="indicators/mte/engine.py",
            producer_fn="compute_mte",
            consumer_module="src/report/volatility_mte.py",
            consumer_fn="render_mte",
            consumer_var="mte_result",
        )

El helper lanza AssertionError con mensaje exacto (que campo anadir
a que funcion, que whitelist existe). Sin silencios: si el AST no
encuentra el patron esperado, falla con AssertionError explicito.
"""
import ast
from pathlib import Path


def _find_function(tree, name):
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name):
            return node
    return None


def _as_path(p):
    if isinstance(p, Path):
        return p
    return Path(p)


def producer_return_keys(module_path, fn_name):
    """Claves del return { ... } de la funcion `fn_name` en module_path.

    Si la funcion tiene varios return { ... }, devuelve las claves de
    la UNION. Es conservador: si alguna rama no incluye el campo,
    el contrato no se cumple. El primer return { ... } encontrado en
    walk no es necesariamente el "principal"; la union evita falsos
    positivos por orden de AST.
    """
    p = _as_path(module_path)
    tree = ast.parse(p.read_text(encoding="utf-8-sig"))
    fn = _find_function(tree, fn_name)
    if fn is None:
        raise AssertionError(
            f"{fn_name} no encontrado en {p}. "
            f"Revisar nombre del modulo o de la funcion."
        )
    keys = set()
    found = False
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            found = True
            for k in node.value.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
    if not found:
        raise AssertionError(
            f"return {{ ... }} no encontrado en {fn_name} ({p}). "
            f"Si la funcion no retorna dict literal, el contrato AST "
            f"no aplica: usar test de integracion normal."
        )
    return keys


def producer_state_keys(module_path, fn_name):
    """Claves del dict de json.dump({...}) dentro de `fn_name`.

    Es la fuente de verdad de lo que se publica en un state JSON
    (outputs/state/*.json). Busca la llamada `json.dump({...}, ...)`
    y extrae las claves del dict.
    """
    p = _as_path(module_path)
    tree = ast.parse(p.read_text(encoding="utf-8-sig"))
    fn = _find_function(tree, fn_name)
    if fn is None:
        raise AssertionError(
            f"{fn_name} no encontrado en {p}."
        )
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
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
    raise AssertionError(
        f"json.dump({{...}}) no encontrado en {fn_name} ({p}). "
        f"Si el state se escribe de otra forma, el contrato AST no "
        f"aplica a esta funcion."
    )


def consumer_read_keys(module_path, fn_name, var_name):
    """Claves que `fn_name` lee via `var_name.get('X')` o `var_name['X']`.

    No detecta lecturas dinamicas (var_name.get(key_variable),
    var_name[expr]) ni unpacking (a, b = var_name['a'], ...). Si
    aparece una lectura dinamica y el contrato falla por falso
    negativo, anadir a mano al whitelist con justificacion.
    """
    p = _as_path(module_path)
    tree = ast.parse(p.read_text(encoding="utf-8-sig"))
    fn = _find_function(tree, fn_name)
    if fn is None:
        raise AssertionError(
            f"{fn_name} no encontrado en {p}."
        )
    keys = set()
    for node in ast.walk(fn):
        # var_name.get('X', ...)
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == var_name):
                if (node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)):
                    keys.add(node.args[0].value)
        # var_name['X']
        if isinstance(node, ast.Subscript):
            if (isinstance(node.value, ast.Name)
                    and node.value.id == var_name
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                keys.add(node.slice.value)
    return keys


def assert_render_subset_return(
        producer_module, producer_fn,
        consumer_module, consumer_fn, consumer_var,
        whitelist=frozenset()):
    """Downstream: consumer lee X => X en producer return.

    Devuelve (reads, returns) para que el test pueda verificar
    regresiones explicitas (p.ej. "futures_status sigue en reads").
    """
    returns = producer_return_keys(producer_module, producer_fn)
    reads = consumer_read_keys(consumer_module, consumer_fn, consumer_var)

    if not reads:
        raise AssertionError(
            f"{consumer_fn} ({consumer_module}) no lee ninguna clave "
            f"de {consumer_var}. El AST no encontro el patron esperado "
            f"({consumer_var}.get o {consumer_var}[...]). Revisar si "
            f"el nombre del parametro cambio o si la lectura es "
            f"dinamica (no soportada por este helper)."
        )

    missing = (reads - whitelist) - returns
    if missing:
        raise AssertionError(
            f"{consumer_fn} lee {sorted(missing)} pero {producer_fn} "
            f"({producer_module}) no los retorna. Clase de bug "
            f"1fb791ab: el campo vive en el state JSON pero no en el "
            f"return, y {consumer_fn} no lo puede pintar. "
            f"Anadir al return de {producer_fn}. "
            f"Return actual: {sorted(returns)}. "
            f"Whitelist (consumer only, justificada): {sorted(whitelist)}."
        )
    return reads, returns


def assert_state_subset_return(
        producer_module, state_fn, return_fn,
        whitelist=frozenset()):
    """Upstream: state publica X (no whitelist) => X en return.

    Detecta campos funcionales que solo viven en el state JSON. La
    whitelist es para metadatos tecnicos (schema_version, fechas,
    cobertura) que no se publican en el reporte por diseno.

    `state_fn` y `return_fn` suelen ser la misma funcion (un compute
    que hace ambos). Si son distintas, se pasan por separado.
    """
    state = producer_state_keys(producer_module, state_fn)
    returns = producer_return_keys(producer_module, return_fn)

    missing = (state - whitelist) - returns
    if missing:
        raise AssertionError(
            f"El state JSON de {state_fn} ({producer_module}) publica "
            f"{sorted(missing)} pero {return_fn} no los incluye. "
            f"Clase 1fb791ab: un campo que solo vive en el state es "
            f"invisible desde el render, aunque el operador pueda "
            f"leerlo del JSON. Anadir al return de {return_fn}. "
            f"State actual: {sorted(state)}. "
            f"Return actual: {sorted(returns)}. "
            f"Whitelist (state only, justificada): {sorted(whitelist)}."
        )
    return state, returns
