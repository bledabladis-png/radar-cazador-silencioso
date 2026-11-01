# -*- coding: utf-8 -*-
"""C.3 (2026-11-01): coherencia entre SHARED_INPUTS y el codigo.

Contrato: cada variable del registro debe aparecer (por token) en la
fuente de cada modulo listado. Si un refactor elimina el uso de una
variable, este test falla y fuerza actualizar el registro (no el
revés: el registro documenta lo que hay).
"""
import pathlib

import pytest

from config.shared_inputs import SHARED_INPUTS, shared_with, variables_in


ROOT = pathlib.Path(__file__).resolve().parent.parent


def _source(module):
    """Resuelve nombre corto a path .py y devuelve texto (o None)."""
    candidates = [
        ROOT / "regimes" / f"{module}.py",
        ROOT / "indicators" / f"{module}.py",
        ROOT / "indicators" / "mte" / f"{module}.py",
        ROOT / "src" / "pipeline" / f"{module}.py",
        ROOT / "src" / "report" / f"{module}.py",
    ]
    for c in candidates:
        if c.exists():
            return c.read_text(encoding="utf-8-sig", errors="replace")
    return None


def _tokens(variable):
    """Tokens a buscar en la fuente para la variable."""
    if variable == "HYG/LQD":
        return ["HYG", "LQD"]
    if variable == "credit_signal":
        return ["credit"]
    return [variable]


@pytest.mark.parametrize("variable", sorted(SHARED_INPUTS.keys()))
def test_variable_en_al_menos_2_modulos(variable):
    """Una variable compartida debe estar en >=2 modulos."""
    mods = shared_with(variable)
    assert len(mods) >= 2, (
        f"Variable {variable!r} aparece en {mods}. "
        f"Si ya no es compartida, retirala del registro."
    )


@pytest.mark.parametrize("variable", sorted(SHARED_INPUTS.keys()))
def test_cada_modulo_menciona_variable(variable):
    """Cada modulo listado debe mencionar el token de la variable."""
    tokens = _tokens(variable)
    for mod in shared_with(variable):
        src = _source(mod)
        assert src is not None, (
            f"Modulo {mod!r} no localizado en disco. "
            f"Registro desfasado o heuristica de _source() incompleta."
        )
        found = any(t in src for t in tokens)
        assert found, (
            f"Variable {variable!r} declarada en modulo {mod!r} pero "
            f"no aparece en su fuente. Tokens probados: {tokens}. "
            f"Registro desfasado (variable retirada o renombrada)."
        )


def test_variables_in_es_inverso_de_shared_with():
    """Coherencia interna: variables_in(shared_with(var)) contiene var."""
    for var, mods in SHARED_INPUTS.items():
        for m in mods:
            assert var in variables_in(m), (
                f"Incoherencia: {var} en {m} pero variables_in({m}) no lo ve."
            )
