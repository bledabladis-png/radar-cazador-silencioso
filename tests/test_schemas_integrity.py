"""Centinela: los .schema trackeados de outputs/history/ existen.

El conftest (`_cleanup_schema_files`) restaura los tracked al teardown
de cada test. Este test cierra el ciclo: si algun dia una regresion
los borra sin restaurar, el CI falla con mensaje claro en lugar de
dejar un D silencioso en git status.
"""
import pathlib
import subprocess


def _tracked_schemas():
    r = subprocess.run(
        ['git', 'ls-files', 'outputs/history/*.schema'],
        capture_output=True, text=True, check=False,
    )
    return [f for f in r.stdout.splitlines() if f.strip()]


def test_todos_los_schemas_trackeados_existen():
    tracked = _tracked_schemas()
    missing = [f for f in tracked if not pathlib.Path(f).exists()]
    assert not missing, (
        f'Esquemas trackeados ausentes: {missing}. '
        'Un test los borro sin restaurar. Ver tests/conftest.py '
        '_cleanup_schema_files.'
    )
