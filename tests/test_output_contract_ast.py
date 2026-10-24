# -*- coding: utf-8 -*-
"""Test de contrato: ningun writer del pipeline escribe a outputs/history/
fuera del wrapper output_writer.persist().

Excepcion unica y justificada:
  - src/pipeline/finalize.py::save_sector_rankings escribe a
    outputs/report/sector_rankings.csv (efimero, gitignored, sin
    dimension temporal). No va al registro por diseno.

El test recorre los .py de src/pipeline/ con ast, detecta llamadas
.to_csv() y falla si aparecen fuera de output_writer.py o de la
funcion save_sector_rankings.
"""
import ast
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent.parent / 'src' / 'pipeline'

# Ficheros donde .to_csv() esta permitido (porque contienen el wrapper
# o porque su escritura es la excepcion documentada).
ALLOWED_FILES = {'output_writer.py'}

# Funciones donde .to_csv() esta permitido (excepcion por diseno).
ALLOWED_FUNCTIONS = {
    # outputs/report/: snapshots efimeros sin dimension temporal,
    # fuera del registro por diseno.
    ('finalize.py', 'save_sector_rankings'),
}


def _find_to_csv_calls(tree):
    """Devuelve lista de (lineno, enclosing_function_name) para cada
    llamada .to_csv() en el arbol."""
    results = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack = []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == 'to_csv':
                enclosing = self.stack[-1] if self.stack else None
                results.append((node.lineno, enclosing))
            self.generic_visit(node)

    V().visit(tree)
    return results


def _iter_pipeline_files():
    for py in sorted(PIPELINE_DIR.glob('*.py')):
        if py.name == '__init__.py':
            continue
        yield py


def test_no_to_csv_outside_wrapper():
    """Ningun fichero del pipeline (excepto output_writer.py y la
    excepcion documentada en finalize.save_sector_rankings) debe llamar
    a .to_csv()."""
    violations = []
    for py in _iter_pipeline_files():
        if py.name in ALLOWED_FILES:
            continue
        tree = ast.parse(py.read_text(encoding='utf-8-sig'), filename=str(py))
        for lineno, func_name in _find_to_csv_calls(tree):
            if (py.name, func_name) in ALLOWED_FUNCTIONS:
                continue
            violations.append(
                f"{py.name}:{lineno} en {func_name or '<module>'}"
            )
    assert not violations, (
        "to_csv() fuera del wrapper. Migrar a output_writer.persist():\n  - "
        + "\n  - ".join(violations)
    )


def test_wrapper_existe():
    """El wrapper debe existir y exponer persist y skip."""
    from src.pipeline import output_writer
    assert hasattr(output_writer, 'persist')
    assert hasattr(output_writer, 'skip')


def test_registry_cubre_writers():
    """Todos los writers del pipeline deben tener su name en el registro.
    Verifica que los imports de persist() usan names validos."""
    from src.pipeline.output_registry import REGISTRY_BY_NAME

    valid_names = set(REGISTRY_BY_NAME.keys())
    used_names = set()
    for py in _iter_pipeline_files():
        if py.name == 'output_writer.py':
            continue
        tree = ast.parse(py.read_text(encoding='utf-8-sig'), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if (isinstance(f, ast.Name)
                        and f.id in ('persist', 'skip')
                        and node.args
                        and isinstance(node.args[0], ast.Constant)):
                    used_names.add(node.args[0].value)

    assert used_names, "ningun writer usa persist/skip"
    unknown = used_names - valid_names
    assert not unknown, f"names no registrados: {sorted(unknown)}"
