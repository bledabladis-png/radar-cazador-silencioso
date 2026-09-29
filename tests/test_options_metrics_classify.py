"""Tests de los clasificadores de options_metrics.

Cubren los guards de finitud de classify_pcr y classify_ihr. Mismo
patron que classify_darkpool (C-08-preventivo, 2026-09-29): cascada
de comparaciones que sin guard mapea NaN/None/inf a un estado real.
"""
from __future__ import annotations

import numpy as np

from indicators.options_metrics import classify_pcr, classify_ihr


# --- classify_pcr ---

def test_pcr_nan_devuelve_sin_historial():
    assert classify_pcr(np.nan) == "Sin historial suficiente"


def test_pcr_none_devuelve_sin_historial():
    assert classify_pcr(None) == "Sin historial suficiente"


def test_pcr_inf_devuelve_sin_historial():
    assert classify_pcr(np.inf) == "Sin historial suficiente"
    assert classify_pcr(-np.inf) == "Sin historial suficiente"


def test_pcr_valores_validos():
    # Cascada completa: cada rama recibe el tipo esperado.
    assert classify_pcr(100.0) == "Pánico"
    assert classify_pcr(-100.0) == "Euforia"


# --- classify_ihr ---

def test_ihr_nan_devuelve_na():
    assert classify_ihr(np.nan) == "N/A"


def test_ihr_none_devuelve_na():
    assert classify_ihr(None) == "N/A"


def test_ihr_valores_validos():
    assert classify_ihr(100.0) == "Cobertura institucional extrema"
    assert classify_ihr(-100.0) == "Especulación extrema"