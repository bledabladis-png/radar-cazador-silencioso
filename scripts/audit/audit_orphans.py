"""Detector refinado de huerfanos - FASE 0.2.

Analisis por reachability desde entry points conocidos:
  - run.py (main productivo)
  - scripts/*.py, tests/*.py, validation/*.py
  - .github/workflows/*.yml referenciando py -m o py scripts/*.py

Read-only. No modifica nada.
Uso: py scripts/audit_orphans.py
"""
from __future__ import annotations

import ast
import re
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "node_modules",
}

def iter_py_files(root: Path):
    for p in root.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        yield p


def module_path_to_dotted(path: Path, root: Path):
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None, False
    if rel.suffix != ".py":
        return None, False
    parts = list(rel.parts)
    is_package = parts[-1] == "__init__.py"
    if is_package:
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    if not parts:
        return None, False
    return ".".join(parts), is_package

def extract_imports(tree, current_module: str, is_package: bool):
    if is_package:
        package = current_module
    elif "." in current_module:
        package = current_module.rsplit(".", 1)[0]
    else:
        package = ""

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                parts = package.split(".") if package else []
                up = node.level - 1
                if up > 0 and up <= len(parts):
                    parts = parts[:-up]
                if node.module:
                    parts = parts + node.module.split(".")
                base = ".".join(parts)
            if base:
                imports.add(base)
                for alias in node.names:
                    if alias.name != "*":
                        imports.add(f"{base}.{alias.name}")
        elif isinstance(node, ast.Call):
            func = node.func
            fname = None
            if isinstance(func, ast.Attribute) and func.attr == "import_module":
                fname = "import_module"
            elif isinstance(func, ast.Name) and func.id == "import_module":
                fname = "import_module"
            if fname and node.args:
                a0 = node.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    imports.add(a0.value)
    return imports

def build_graph(root, files):
    graph = {}
    for p in files:
        mod, is_pkg = module_path_to_dotted(p, root)
        if not mod:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8-sig", errors="replace"))
            graph[mod] = extract_imports(tree, mod, is_pkg)
        except Exception:
            graph[mod] = set()
    return graph


def reachable_from(graph, roots):
    seen = set()
    q = deque(roots)
    while q:
        m = q.popleft()
        if m in seen:
            continue
        seen.add(m)
        parts = m.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in seen:
                q.append(parent)
        for dep in graph.get(m, set()):
            if dep not in seen:
                q.append(dep)
    return seen

def scan_workflow_refs(root):
    refs = set()
    wf_dir = root / ".github" / "workflows"
    if not wf_dir.exists():
        return refs
    mod_re = re.compile(r"py\s+-m\s+([A-Za-z_][A-Za-z0-9_.]*)")
    script_re = re.compile(r"py\s+(scripts[\\/][\w\\/.-]+\.py)")
    for yml in wf_dir.glob("*.yml"):
        try:
            text = yml.read_text(encoding="utf-8-sig", errors="replace")
        except Exception:
            continue
        for m in mod_re.finditer(text):
            refs.add(m.group(1))
        for m in script_re.finditer(text):
            rel = m.group(1).replace("\\", "/")
            dotted = rel[:-3].replace("/", ".")
            refs.add(dotted)
    return refs


def main():
    files = list(iter_py_files(ROOT))
    graph = build_graph(ROOT, files)

    roots = set()
    if (ROOT / "run.py").exists():
        roots.add("run")
    for p in files:
        rel = p.relative_to(ROOT)
        top = rel.parts[0]
        if top in ("scripts", "tests", "validation"):
            mod, _ = module_path_to_dotted(p, ROOT)
            if mod:
                roots.add(mod)
    roots |= scan_workflow_refs(ROOT)

    reachable = reachable_from(graph, roots)

    prod_prefixes = ("src.", "indicators.", "regimes.", "config.")
    all_prod = set()
    for mod in graph:
        for pref in prod_prefixes:
            if mod.startswith(pref):
                all_prod.add(mod)
                break

    orphans = sorted(all_prod - reachable)
    print(f"=== Huerfanos reales (no alcanzables): {len(orphans)} ===")
    for o in orphans:
        print(f"  {o}")

    reachable_prod = sorted(all_prod & reachable)
    print(f"\n=== Modulos de produccion alcanzables: {len(reachable_prod)} ===")
    for r in reachable_prod:
        print(f"  {r}")


if __name__ == "__main__":
    main()