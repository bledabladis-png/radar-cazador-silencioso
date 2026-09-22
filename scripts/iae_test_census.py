"""Censo riguroso de tests del modulo IAE.

Determina que ficheros tests/test_*.py importan src.institutional_accumulation
(por AST, no por nombre), y cuenta tests por tres metodos:
  - pytest --collect-only (expandido, cuenta casos parametrize)
  - AST de funciones def test_ (sin expandir)
  - Casos parametrize aproximados

Uso: py scripts/iae_test_census.py
Salida: outputs/audit/iae_test_census/census.json + resumen por stdout.

Criterio canonico del modulo IAE:
  tests en ficheros tests/test_*.py que importan
  src.institutional_accumulation (verificado por AST).

Este criterio es determinista y reproducible. Sustituye al conteo
historico "641" (no reproducible con ningun criterio mecanico).
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
OUT = ROOT / "outputs" / "audit" / "iae_test_census"


def imports_iae(py_path: Path) -> tuple[bool, list[str]]:
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False, []
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if "institutional_accumulation" in a.name:
                    hits.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if "institutional_accumulation" in mod:
                hits.append(mod)
    return len(hits) > 0, hits


def git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    all_test_files = sorted(TESTS.glob("test_*.py"))
    iae_files = []
    for f in all_test_files:
        ok, hits = imports_iae(f)
        if ok:
            iae_files.append({"file": f.name, "imports": hits})

    if not args.quiet:
        print("=" * 72)
        print("CENSO DE TESTS DEL MODULO IAE")
        print("=" * 72)
        print("Ficheros tests/ totales: " + str(len(all_test_files)))
        print("Ficheros que importan src.institutional_accumulation: "
              + str(len(iae_files)))

    per_file = []
    total_pytest = 0
    for entry in iae_files:
        p = TESTS / entry["file"]
        r = subprocess.run(
            [sys.executable, "-m", "pytest", str(p), "--collect-only", "-q"],
            capture_output=True, text=True, cwd=str(ROOT))
        line = ""
        for ln in (r.stdout or "").splitlines():
            if "tests collected" in ln or "test collected" in ln:
                line = ln.strip()
                break
        n = 0
        if line:
            try:
                n = int(line.split("/")[-1].split()[0])
            except Exception:
                n = 0
        per_file.append({"file": entry["file"], "pytest_collected": n})
        total_pytest += n

    per_file_ast = []
    total_funcs = 0
    total_params = 0
    for entry in iae_files:
        p = TESTS / entry["file"]
        try:
            tree = ast.parse(p.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        funcs = [n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
        n_funcs = len(funcs)
        n_params = 0
        for fn in funcs:
            for dec in fn.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                    if dec.func.attr == "parametrize" and len(dec.args) >= 2:
                        arg = dec.args[1]
                        if isinstance(arg, ast.List):
                            n_params += len(arg.elts)
        per_file_ast.append({
            "file": entry["file"],
            "n_test_functions": n_funcs,
            "n_parametrize_cases_approx": n_params,
        })
        total_funcs += n_funcs
        total_params += n_params

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "criterio_inclusion": "importa src.institutional_accumulation (AST)",
        "n_ficheros_totales_tests": len(all_test_files),
        "n_ficheros_modulo_iae": len(iae_files),
        "total_pytest_collected": total_pytest,
        "total_funcs_ast": total_funcs,
        "total_parametrize_cases_approx": total_params,
        "per_file": per_file,
        "per_file_ast": per_file_ast,
        "files": [e["file"] for e in iae_files],
    }
    out = OUT / "census.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.quiet:
        print()
        print("RESUMEN")
        print("  Ficheros del modulo IAE       : " + str(len(iae_files)))
        print("  Tests collected (pytest)      : " + str(total_pytest))
        print("  Funciones test_ (AST)         : " + str(total_funcs))
        print("  Casos parametrize (aprox)     : " + str(total_params))
        print("  JSON: " + str(out))

    return 0


if __name__ == "__main__":
    sys.exit(main())