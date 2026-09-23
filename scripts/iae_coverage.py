"""Cobertura de tests del modulo IAE (criterio AST canonico).

Resuelve por AST los ficheros tests/test_*.py que importan
src.institutional_accumulation (mismo criterio que
scripts/iae_test_census.py) y ejecuta pytest con
--cov=src/institutional_accumulation sobre el universo completo.

Resultado esperado a 2026-09-23: 759 passed, 92% coverage.
"""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
TARGET = "institutional_accumulation"


def find_test_files():
    files = []
    for f in sorted(TESTS.glob("test_*.py")):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        for node in ast.walk(tree):
            hit = False
            if isinstance(node, ast.Import):
                if any(TARGET in a.name for a in node.names):
                    hit = True
            elif isinstance(node, ast.ImportFrom):
                if TARGET in (node.module or ""):
                    hit = True
            if hit:
                files.append(str(f))
                break
    return files


def main():
    files = find_test_files()
    print(f"Ficheros del modulo IAE: {len(files)}")
    if len(files) != 42:
        print(f"WARN: se esperaban 42 ficheros, encontrados {len(files)}")
    cmd = [sys.executable, "-m", "pytest"] + files + [
        "-q", "--tb=line",
        "--cov=src/institutional_accumulation",
        "--cov-report=term",
    ]
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


if __name__ == "__main__":
    sys.exit(main())