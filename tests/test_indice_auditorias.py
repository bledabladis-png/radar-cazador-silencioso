# -*- coding: utf-8 -*-
"""Test del indice consolidado H1-H18 (Fase 5)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.genera_indice_auditorias as gi


def test_indice_existe():
    assert gi.OUT.exists(), (
        "INDEX_H1_H18.md no existe. Ejecutar: "
        "py scripts/genera_indice_auditorias.py --apply"
    )


def test_indice_actualizado():
    """El fichero versionado debe coincidir con lo que genera el script."""
    esperado = gi.generar()
    actual = gi.OUT.read_text(encoding="utf-8")
    assert actual == esperado, (
        "el indice esta desactualizado. Ejecutar: "
        "py scripts/genera_indice_auditorias.py --apply"
    )


def test_indice_incluye_h2_false_positive():
    text = gi.OUT.read_text(encoding="utf-8")
    lineas = [ln for ln in text.splitlines() if ln.startswith("| H2 |")]
    assert lineas, "H2 no aparece en el indice"
    assert "FALSE_POSITIVE" in lineas[0]


def test_indice_incluye_h7_fixed():
    text = gi.OUT.read_text(encoding="utf-8")
    lineas = [ln for ln in text.splitlines() if ln.startswith("| H7 |")]
    assert lineas, "H7 no aparece en el indice"
    assert "FIXED" in lineas[0]
    assert "1189b7f8" in lineas[0]


def test_indice_incluye_los_18():
    """Los 18 hallazgos H1..H18 estan presentes."""
    text = gi.OUT.read_text(encoding="utf-8")
    for n in range(1, 19):
        assert "| H{0} |".format(n) in text, "falta H{0}".format(n)
