"""Tests de los post-procesadores N-PORT (F5.7-19). Sin red."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load(name, relpath):
    p = ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- sec_nport_quarters_position_change.py ----------

def test_snqpc_no_hardcode_quarters():
    p = ROOT / "data" / "providers" / "sec_nport_quarters_position_change.py"
    src = p.read_text(encoding="utf-8-sig")
    assert "QUARTERS = [" not in src
    assert "2026q1" not in src
    assert "2026q2" not in src


def test_snqpc_tiene_discover_quarters():
    p = ROOT / "data" / "providers" / "sec_nport_quarters_position_change.py"
    tree = ast.parse(p.read_text(encoding="utf-8-sig"))
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "discover_quarters" in funcs


def test_discover_quarters_funciona(tmp_path):
    mod = _load("snqpc_test", "data/providers/sec_nport_quarters_position_change.py")
    for name in ["2025q4", "2026q1", "2026q2"]:
        (tmp_path / name).mkdir()
    (tmp_path / "README").write_text("x")
    (tmp_path / "otros").mkdir()
    result = mod.discover_quarters(base=tmp_path)
    assert result == ["2025q4", "2026q1", "2026q2"]


def test_discover_quarters_vacio(tmp_path):
    mod = _load("snqpc_test2", "data/providers/sec_nport_quarters_position_change.py")
    empty = tmp_path / "vacio"
    empty.mkdir()
    assert mod.discover_quarters(base=empty) == []
    assert mod.discover_quarters(base=tmp_path / "no_existe") == []


# ---------- qqq_nport_flow.py ----------

def test_qqq_no_hardcode_xml_path():
    p = ROOT / "data" / "providers" / "qqq_nport_flow.py"
    src = p.read_text(encoding="utf-8-sig")
    assert "nport_2026-03-31.xml" not in src
    assert "XML_PATH" not in src


def test_qqq_tiene_funciones_discover():
    p = ROOT / "data" / "providers" / "qqq_nport_flow.py"
    tree = ast.parse(p.read_text(encoding="utf-8-sig"))
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "find_latest_xml" in funcs
    assert "parse_report_date_from_xml_name" in funcs


def test_find_latest_xml_vacio(tmp_path):
    mod = _load("qqq_test", "data/providers/qqq_nport_flow.py")
    assert mod.find_latest_xml(cache_dir=tmp_path) is None
    assert mod.find_latest_xml(cache_dir=tmp_path / "no_existe") is None


def test_find_latest_xml_con_varios(tmp_path):
    mod = _load("qqq_test2", "data/providers/qqq_nport_flow.py")
    (tmp_path / "nport_2026-01-01.xml").write_text("<x/>")
    (tmp_path / "nport_2026-03-31.xml").write_text("<x/>")
    (tmp_path / "otro.xml").write_text("<x/>")
    result = mod.find_latest_xml(cache_dir=tmp_path)
    assert result is not None
    assert result.name == "nport_2026-03-31.xml"


def test_parse_report_date_from_xml_name():
    mod = _load("qqq_test3", "data/providers/qqq_nport_flow.py")
    p = Path("data/cache/sec/qqq/nport_2026-03-31.xml")
    assert mod.parse_report_date_from_xml_name(p) == "2026-03-31"
    bad = Path("data/cache/sec/qqq/otro.xml")
    assert mod.parse_report_date_from_xml_name(bad) is None
