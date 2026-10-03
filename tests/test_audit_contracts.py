# -*- coding: utf-8 -*-
"""Contratos mecanicos derivados de S-03-deep.

1. test_dict_keys_alineado: docstring "Returns: dict con keys:"
   debe coincidir con las keys del return dict literal.

2. test_no_11_hardcoded_en_sectorial: prohibido Constant(11) en
   contextos sectoriales (Compare, BinOp *, /, +, -, Assign a
   variable con nombre sectorial, slice [:11]) en las 3 rutas
   auditadas de S-03-deep.

Si fallan, corregir el codigo o actualizar el docstring. No relajar
estos tests sin decision documentada.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROOTS_AUDITADAS = [ROOT / "src" / "pipeline", ROOT / "regimes", ROOT / "src" / "report"]
SETTINGS = ROOT / "config" / "settings.py"


def _iter_py_files(roots):
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            yield p


# ---------------------------------------------------------------------------
# Test 1: docstring "dict con keys" alineado con return
# ---------------------------------------------------------------------------

def _tokenize_keys_line(line):
    """Devuelve tokens de una linea de keys, cortando por comas."""
    out = []
    for part in line.split(","):
        part = part.strip()
        if not part:
            continue
        token = part.split("(")[0].split(":")[0].strip()
        if not token:
            continue
        if token[0].isalpha() or token[0] == "_":
            if all(c.isalnum() or c == "_" for c in token):
                out.append(token)
    return out


def _extract_declared_keys(docstring):
    """Keys declaradas en el bloque 'Returns: dict con keys:'."""
    if not docstring or "dict con keys" not in docstring:
        return None
    keys = set()
    in_block = False
    for line in docstring.split("\n"):
        s = line.strip()
        if "dict con keys" in s:
            in_block = True
            continue
        if not in_block:
            continue
        if not s:
            continue
        if s.startswith(("Returns", "Args", "Raises")):
            break
        keys.update(_tokenize_keys_line(s))
    return keys or None


def _extract_return_keys(fn):
    """Keys del primer return{...} literal encontrado en la funcion."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys = set()
            for k in node.value.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
            if keys:
                return keys
    return None


def _collect_dict_key_mismatches():
    mismatches = []
    for p in _iter_py_files(ROOTS_AUDITADAS):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            declared = _extract_declared_keys(ast.get_docstring(node))
            if not declared:
                continue
            actual = _extract_return_keys(node)
            if not actual:
                continue
            if declared != actual:
                mismatches.append((p, node.name, declared, actual))
    return mismatches


def test_dict_keys_alineado():
    """Docstring 'Returns: dict con keys:' == keys del return dict."""
    mismatches = _collect_dict_key_mismatches()
    if not mismatches:
        return
    lines = ["Docstring y return desalineados:"]
    for p, fn, declared, actual in mismatches:
        rel = p.relative_to(ROOT)
        extra_dec = sorted(declared - actual)
        extra_ret = sorted(actual - declared)
        lines.append(f"  {rel}:{fn}")
        if extra_dec:
            lines.append(f"    docstring declara pero no devuelve: {extra_dec}")
        if extra_ret:
            lines.append(f"    devuelve pero docstring no declara: {extra_ret}")
    pytest.fail("\n".join(lines))


# ---------------------------------------------------------------------------
# Test 2: no Constant(11) hardcoded en contextos sectoriales
# ---------------------------------------------------------------------------

_SECTOR_NAME_HINTS = ("total", "sector", "n_", "num_sectores", "expected_sector")


def _is_sectorish_name(name):
    low = name.lower()
    return any(h in low for h in _SECTOR_NAME_HINTS)


def _find_11_hardcoded(tree):
    """Devuelve lista de (node, motivo) para Constant(11) sospechosos."""
    found = []

    def _is_11(node):
        return isinstance(node, ast.Constant) and node.value == 11

    for node in ast.walk(tree):
        # Compare: x == 11, x != 11, etc.
        if isinstance(node, ast.Compare):
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.Gt,
                                   ast.LtE, ast.GtE)) and _is_11(comp):
                    found.append((node, "Compare"))
                if isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.Gt,
                                   ast.LtE, ast.GtE)) and _is_11(node.left):
                    found.append((node, "Compare"))

        # BinOp con * / + -
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Mult, ast.Div, ast.Add, ast.Sub)):
                if _is_11(node.left) or _is_11(node.right):
                    found.append((node, "BinOp"))

        # Assign: name = 11 con nombre sectorial
        if isinstance(node, ast.Assign) and _is_11(node.value):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and _is_sectorish_name(tgt.id):
                    found.append((node, f"Assign({tgt.id})"))
        if isinstance(node, ast.AnnAssign) and _is_11(node.value):
            if isinstance(node.target, ast.Name) and _is_sectorish_name(node.target.id):
                found.append((node, f"AnnAssign({node.target.id})"))

        # Slice [11] literal (solo en subscripts, no en range)
        if isinstance(node, ast.Subscript):
            sl = node.slice
            if isinstance(sl, ast.Constant) and sl.value == 11:
                found.append((node, "Subscript[11]"))
            if isinstance(sl, ast.Slice):
                if isinstance(sl.upper, ast.Constant) and sl.upper.value == 11:
                    found.append((node, "Slice[:11]"))

    return found


def _collect_11_hardcoded():
    offenders = []
    for p in _iter_py_files(ROOTS_AUDITADAS):
        if p.resolve() == SETTINGS.resolve():
            continue
        try:
            src = p.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError):
            continue
        found = _find_11_hardcoded(tree)
        if found:
            offenders.append((p, found))
    return offenders


def test_no_11_hardcoded_en_sectorial():
    """Sin Constant(11) en contextos sectoriales en las rutas auditadas."""
    offenders = _collect_11_hardcoded()
    if not offenders:
        return
    lines = ["Constant(11) hardcoded en contexto sectorial:"]
    for p, found in offenders:
        rel = p.relative_to(ROOT)
        for node, motivo in found:
            lines.append(f"  {rel}:{node.lineno} {motivo} -> {ast.dump(node)[:80]}")
    lines.append("Usar EXPECTED_SECTOR_COUNT de config.settings.")
    pytest.fail("\n".join(lines))

# ---------------------------------------------------------------------------
# Auto-verificacion: el detector detecta lo que dice detectar
# ---------------------------------------------------------------------------

def test_detector_11_detecta_assign_sectorial():
    """n_total = 11 debe ser detectado."""
    tree = ast.parse("def f():\n    n_total = 11\n")
    found = _find_11_hardcoded(tree)
    assert len(found) == 1, f"esperado 1 hallazgo, hay {len(found)}"


def test_detector_11_detecta_binop():
    """x * 11 debe ser detectado."""
    tree = ast.parse("def f():\n    return n * 11\n")
    found = _find_11_hardcoded(tree)
    assert len(found) == 1


def test_detector_11_ignora_range_y_kwargs():
    """range(11) y llamadas con 11 no son contexto sectorial."""
    tree = ast.parse("def f():\n    for i in range(11):\n        pass\n")
    found = _find_11_hardcoded(tree)
    assert len(found) == 0, f"falso positivo: {found}"


def test_detector_keys_detecta_mismatch():
    """Docstring 3 keys + return 4 keys debe dar mismatch."""
    src = (
        "def f():\n"
        "    \"\"\"x\n"
        "    Returns:\n"
        "        dict con keys:\n"
        "            a, b, c\n"
        "    \"\"\"\n"
        "    return {\"a\": 1, \"b\": 2, \"c\": 3, \"d\": 4}\n"
    )
    tree = ast.parse(src)
    fn = tree.body[0]
    declared = _extract_declared_keys(ast.get_docstring(fn))
    actual = _extract_return_keys(fn)
    assert declared != actual
    assert actual - declared == {"d"}
