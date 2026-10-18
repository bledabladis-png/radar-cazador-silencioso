"""Verifica que generate_estado_sistema.py no invoca validation/ en pytest.

Test estatico (AST): inspecciona el codigo fuente, no ejecuta main().
Sobrevive a refactors de la firma de section_tests.
"""
import ast
from pathlib import Path

SRC = Path(__file__).parent.parent / "scripts" / "generate_estado_sistema.py"


def test_pytest_call_no_incluye_validation_dir():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "run"):
            continue
        if not node.args or not isinstance(node.args[0], ast.List):
            continue
        elts = [e.value for e in node.args[0].elts if isinstance(e, ast.Constant)]
        if "pytest" in elts:
            assert "validation/" not in elts, f"cmd contiene validation/: {elts}"
            assert "tests/" in elts, f"cmd no contiene tests/: {elts}"
            found = True
    assert found, "no se encontro llamada run([...pytest...]) en el fuente"