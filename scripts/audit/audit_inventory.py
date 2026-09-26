"""Auditoria de inventario - FASE 0 Gate 0.

Read-only. No modifica nada.
Uso: py scripts/audit_inventory.py
"""
from __future__ import annotations

import ast
from collections import defaultdict
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


def count_loc(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return 0
    return len(text.splitlines())


def extract_imports(path: Path) -> set:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"))
    except Exception:
        return set()
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                mods.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.add(node.module.split(".")[0])
    return mods

def file_inventory(root: Path) -> dict:
    by_top = defaultdict(lambda: {"files": 0, "loc": 0})
    for p in iter_py_files(root):
        rel = p.relative_to(root)
        if len(rel.parts) == 1:
            top = "(root)"
        elif rel.parts[0] == "src" and len(rel.parts) >= 2:
            top = "src/" + rel.parts[1]
        else:
            top = rel.parts[0]
        by_top[top]["files"] += 1
        by_top[top]["loc"] += count_loc(p)
    return dict(by_top)


def module_path_to_dotted(path: Path, root: Path):
    rel = path.relative_to(root)
    if rel.suffix != ".py":
        return None
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)

def find_orphans(root, files, all_text):
    orphans = []
    for p in files:
        if p.name == "__init__.py":
            continue
        rel = p.relative_to(root)
        if rel.parts[0] in ("tests", "scripts", "validation", "docs"):
            continue
        dotted = module_path_to_dotted(p, root)
        if not dotted:
            continue
        found = False
        for q, text in all_text.items():
            if q == p:
                continue
            if dotted in text:
                found = True
                break
        if not found:
            orphans.append(dotted)
    return orphans


def find_duplicate_basenames(files):
    by_name = defaultdict(list)
    for p in files:
        by_name[p.name].append(str(p))
    return {k: v for k, v in by_name.items() if len(v) > 1}

LOCAL_TOP = {"src", "indicators", "regimes", "config", "validation"}


def build_import_graph(root, files):
    graph = {}
    for p in files:
        mod = module_path_to_dotted(p, root)
        if not mod:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8-sig", errors="replace"))
        except Exception:
            graph[mod] = set()
            continue
        deps = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top in LOCAL_TOP:
                    deps.add(node.module)
            elif isinstance(node, ast.Import):
                for n in node.names:
                    top = n.name.split(".")[0]
                    if top in LOCAL_TOP:
                        deps.add(n.name)
        graph[mod] = deps
    return graph


def find_cycles(graph):
    cycles = []
    visited = set()

    def dfs(node, path):
        if node in path:
            idx = path.index(node)
            cycles.append(path[idx:] + [node])
            return
        if node in visited:
            return
        visited.add(node)
        path.append(node)
        for nxt in graph.get(node, set()):
            dfs(nxt, path)
        path.pop()

    for node in graph:
        dfs(node, [])
    return cycles

def main():
    files = sorted(iter_py_files(ROOT))
    all_text = {}
    for p in files:
        try:
            all_text[p] = p.read_text(encoding="utf-8-sig", errors="replace")
        except Exception:
            all_text[p] = ""

    inv = file_inventory(ROOT)
    print("=== INVENTARIO POR CAPA ===")
    for k in sorted(inv):
        print(f"{k:35s} files={inv[k]['files']:4d} loc={inv[k]['loc']:6d}")

    orphans = find_orphans(ROOT, files, all_text)
    print(f"\n=== POSIBLES HUERFANOS ({len(orphans)}) ===")
    for o in orphans:
        print(f"  {o}")

    dups = find_duplicate_basenames(files)
    print(f"\n=== BASENAMES DUPLICADOS ({len(dups)}) ===")
    for k, v in sorted(dups.items()):
        print(f"  {k}: {len(v)} ocurrencias")
        for path in v:
            print(f"    {path}")

    graph = build_import_graph(ROOT, files)
    cycles = find_cycles(graph)
    print(f"\n=== CICLOS DE IMPORT DETECTADOS ({len(cycles)}) ===")
    for c in cycles[:30]:
        print("  " + " -> ".join(c))
    if len(cycles) > 30:
        print(f"  ... y {len(cycles) - 30} mas")


if __name__ == "__main__":
    main()
