"""Contrato de funciones puras en scripts SEC/QQQ.

Cubre la logica sin red ni filesystem de update_qqq_sec_flow.py y
update_sec_nport_data.py. El resto de los 2 scripts es HTTP, glob o
subprocess; no se testea por ROI < 1.

Complementa test_provider_nport.py (que cubre discover_quarters,
find_latest_xml y parse_report_date_from_xml_name).
"""
from __future__ import annotations

import importlib.util
from datetime import date as real_date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def _load(name, relpath):
    p = ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------
# update_qqq_sec_flow.parse_number
# ---------------------------------------------------------------------

def _parse_number():
    return _load("uqsf_test", "scripts/update_qqq_sec_flow.py").parse_number


def test_parse_number_none_and_empty():
    fn = _parse_number()
    assert fn(None) is None
    assert fn("") is None
    assert fn("   ") is None
    assert fn("nan") is None
    assert fn("none") is None
    assert fn("-") is None
    assert fn(")") is None


def test_parse_number_currency_and_commas():
    fn = _parse_number()
    assert fn("$1,234.56") == 1234.56
    assert fn("1,234.56") == 1234.56
    assert fn("1 234.56") == 1234.56


def test_parse_number_parentheses_negatives():
    fn = _parse_number()
    assert fn("(1.5)") == -1.5
    assert fn("(1.5") == -1.5
    assert fn("(1,234.56)") == -1234.56


def test_parse_number_invalid_returns_none():
    fn = _parse_number()
    assert fn("abc") is None
    assert fn("1.2.3") is None


# ---------------------------------------------------------------------
# update_qqq_sec_flow.normalize_label
# ---------------------------------------------------------------------

def test_normalize_label():
    mod = _load("uqsf_test2", "scripts/update_qqq_sec_flow.py")
    assert mod.normalize_label(None) == ""
    assert mod.normalize_label("  Hello  World  ") == "hello world"
    assert mod.normalize_label("Item (B.6)") == "item b.6"
    # non-breaking space
    assert mod.normalize_label("A\u00a0B") == "a b"


def test_find_label_row():
    mod = _load("uqsf_test3", "scripts/update_qqq_sec_flow.py")
    df = pd.DataFrame([["a", "b"], ["c", "d"]], columns=["x", "y"])
    assert mod.find_label_row(df, "C") == 1
    assert mod.find_label_row(df, "  C  ") == 1
    assert mod.find_label_row(df, "z") is None


# ---------------------------------------------------------------------
# update_sec_nport_data: quarters puros
# ---------------------------------------------------------------------

class _FakeDate(real_date):
    _today = real_date(2026, 1, 1)

    @classmethod
    def today(cls):
        return cls._today


def _usnd():
    return _load("usnd_test", "scripts/update_sec_nport_data.py")


def _patch_today(monkeypatch, mod, y, m, d):
    _FakeDate._today = real_date(y, m, d)
    monkeypatch.setattr(mod.datetime, "date", _FakeDate)


def test_get_last_closed_quarter_por_mes(monkeypatch):
    mod = _usnd()
    cases = [
        (1, "2025q4"), (2, "2025q4"), (3, "2025q4"),
        (4, "2026q1"), (5, "2026q1"), (6, "2026q1"),
        (7, "2026q2"), (8, "2026q2"), (9, "2026q2"),
        (10, "2026q3"), (11, "2026q3"), (12, "2026q3"),
    ]
    for month, expected in cases:
        _patch_today(monkeypatch, mod, 2026, month, 15)
        assert mod.get_last_closed_quarter() == expected, (month, expected)


def test_previous_quarter():
    mod = _usnd()
    assert mod.previous_quarter("2026q1") == "2025q4"
    assert mod.previous_quarter("2026q2") == "2026q1"
    assert mod.previous_quarter("2026q3") == "2026q2"
    assert mod.previous_quarter("2026q4") == "2026q3"
    assert mod.previous_quarter("2024q1") == "2023q4"


def test_quarters_range():
    mod = _usnd()
    assert mod._quarters_range("2026q2", 0) == ["2026q2"]
    assert mod._quarters_range("2026q2", 1) == ["2026q1", "2026q2"]
    assert mod._quarters_range("2026q2", 3) == ["2025q3", "2025q4", "2026q1", "2026q2"]
    # Frontera de anio
    assert mod._quarters_range("2026q1", 2) == ["2025q3", "2025q4", "2026q1"]
