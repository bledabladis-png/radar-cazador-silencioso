# -*- coding: utf-8 -*-
"""Tests de regresion B2 (criterio de seleccion de lideres).

Contexto: el render "Acciones Seleccionadas por el Modelo de Liderazgo
Sectorial" no declaraba como se eligen los 5 lideres por sector. B2
anade una nota que describe los dos pasos (peso en ETF -> WLS).
"""

from src.report.leaders import render_acciones_seleccionadas


def test_b2_nota_criterio_aparece_con_datos():
    """Con leader_lines no vacio, la nota de criterio debe aparecer."""
    out = render_acciones_seleccionadas(["| AAPL | ... |\n"])
    body = "".join(out)
    assert "Criterio de seleccion" in body
    assert "WLS" in body


def test_b2_nota_criterio_no_aparece_sin_datos():
    """Sin leader_lines, la rama vacia no muestra la nota (no hay lideres)."""
    out = render_acciones_seleccionadas(None)
    body = "".join(out)
    assert "Criterio de seleccion" not in body
    assert "No disponibles" in body


def test_b2_titulo_sin_cambios():
    """B2 no cambia el titulo; sigue siendo 'Acciones Seleccionadas'."""
    out = render_acciones_seleccionadas(["| AAPL | ... |\n"])
    body = "".join(out)
    assert "Acciones Seleccionadas por el Modelo de Liderazgo Sectorial" in body


def test_b2_leader_lines_presentes():
    """Las lineas de lideres pasan integras al output."""
    lineas = ["| AAPL | 1.5 |\n", "| MSFT | 1.2 |\n"]
    out = render_acciones_seleccionadas(lineas)
    body = "".join(out)
    assert "AAPL" in body
    assert "MSFT" in body