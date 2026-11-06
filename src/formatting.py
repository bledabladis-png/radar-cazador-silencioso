# -*- coding: utf-8 -*-
"""Formateo numerico determinista para el reporte.

Centraliza el formato de floats con signo forzado y evita el
artefacto "-0.00" (negativo cero): ocurre cuando el valor redondeado
es cero pero el signo original era negativo (ej. -0.001).

Reglas:
  - None, NaN, +-inf -> "N/A"
  - Redondeo a N decimales ANTES de decidir el signo.
  - Cero (positivo o negativo) -> "+0.00" (nunca "-0.00").
  - Resto -> formato con signo explicito (+/-).

Fuente: bug REPORT-NEGATIVE-ZERO (S30, 2026-11-06).
"""
from __future__ import annotations

import math


def format_signed_float(value, decimals: int = 2, na: str = "N/A") -> str:
    """Formatea un float con signo forzado, sin negativo cero.

    Args:
        value: numero, o None.
        decimals: decimales a mostrar (>=0).
        na: texto para None/NaN/inf.

    Returns:
        str con signo explicito. "-0.00" nunca se produce.
    """
    if value is None:
        return na
    try:
        f = float(value)
    except (TypeError, ValueError):
        return na
    if not math.isfinite(f):
        return na
    rounded = round(f, decimals)
    if rounded == 0.0:
        rounded = 0.0
    return f"{rounded:+.{decimals}f}"
