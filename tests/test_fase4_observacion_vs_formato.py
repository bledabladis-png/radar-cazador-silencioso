"""Tests de Fase 4 P7: separacion observacion vs formato.

Fase 4 P7 (2026-10-28). Dos funciones en src/utils.py:

- resolve_reference_date(reference_date): fallback a now() si None.
  Para funciones de FORMATO/render. El timestamp del texto
  renderizado puede ser "ahora" sin romper determinismo de datos.

- resolve_observation_date(reference_date): raise si None. Para
  funciones de OBSERVACION (output persistente en
  outputs/history, outputs/state, data/*.parquet, data/*.csv).

Aplicadas a: flows_secondary (flows_secondary.csv), mte_confirmation
(mte_state.json), validation_gate (validation_gate_result.json).
"""
from __future__ import annotations

import ast
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import resolve_observation_date, resolve_reference_date


# --- resolve_observation_date ---

def test_observation_date_raise_si_none():
    with pytest.raises(ValueError, match="reference_date=None"):
        resolve_observation_date(None)


def test_observation_date_acepta_naive():
    ref = datetime(2026, 10, 27, 12, 0, 0)
    assert resolve_observation_date(ref) == ref
    assert resolve_observation_date(ref).tzinfo is None


def test_observation_date_normaliza_tz():
    ref = datetime(2026, 10, 27, 12, 0, 0, tzinfo=ZoneInfo("Europe/Madrid"))
    out = resolve_observation_date(ref)
    assert out.tzinfo is None
    assert out == datetime(2026, 10, 27, 12, 0, 0)


# --- resolve_reference_date (sigue con fallback) ---

def test_reference_date_fallback_si_none():
    out = resolve_reference_date(None)
    assert isinstance(out, datetime)
    assert out.tzinfo is None  # naive, coherente con el pre-refactor


def test_reference_date_acepta_naive():
    ref = datetime(2026, 10, 27)
    assert resolve_reference_date(ref) == ref


def test_reference_date_normaliza_tz():
    ref = datetime(2026, 10, 27, tzinfo=ZoneInfo("Europe/Madrid"))
    out = resolve_reference_date(ref)
    assert out.tzinfo is None


# --- Guard AST: observacion usa observation, formato usa reference ---

OBSERVACION_MODULOS = (
    # (ruta, funcion que debe usar resolve_observation_date)
    ("src/pipeline/flows_secondary.py", "compute_flows_secondary"),
    ("src/pipeline/mte_confirmation.py", "_compute_mte"),
    ("src/pipeline/validation_gate.py", "run_validation_gate"),
)


@pytest.mark.parametrize("rel,func", OBSERVACION_MODULOS)
def test_observacion_usa_resolve_observation_date(rel, func):
    """Las funciones de observacion NO deben usar resolve_reference_date
    (fallback a now). Deben usar resolve_observation_date (raise)."""
    src = (ROOT / rel).read_text(encoding="utf-8-sig", errors="replace")
    tree = ast.parse(src)
    # Localizar la funcion
    encontrado = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            # Dentro de esta funcion, buscar resolve_reference_date
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    nombre = None
                    if isinstance(sub.func, ast.Name):
                        nombre = sub.func.id
                    elif isinstance(sub.func, ast.Attribute):
                        nombre = sub.func.attr
                    assert nombre != "resolve_reference_date", (
                        f"{rel}:{sub.lineno} ({func}) usa "
                        f"resolve_reference_date. Es observacion, debe "
                        f"usar resolve_observation_date."
                    )
            encontrado = True
            break
    assert encontrado, f"{func} no encontrada en {rel}"


def test_formato_sigue_usando_reference_date():
    """Sanity: las funciones de formato siguen usando
    resolve_reference_date (fallback). Asi el test
    test_reference_date_none_fallback de test_render_freshness_d36.py
    sigue teniendo sentido."""
    src = (ROOT / "src" / "report" / "freshness.py").read_text(
        encoding="utf-8-sig", errors="replace"
    )
    assert "resolve_reference_date" in src, (
        "freshness.py deberia seguir usando resolve_reference_date. "
        "Si se migro a resolve_observation_date, hay que actualizar "
        "test_render_freshness_d36.py::test_reference_date_none_fallback."
    )
