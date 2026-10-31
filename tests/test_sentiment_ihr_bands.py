# -*- coding: utf-8 -*-
"""Centinela H5.3: las bandas IHR del reporte se derivan de los thresholds."""

from config.settings import IHR_THRESHOLDS
from indicators.options_metrics import IHR_LABELS
from src.report.helpers import _fmt_ihr_bands


def test_bandas_citan_todos_los_thresholds():
    """Cada valor de IHR_THRESHOLDS aparece en el texto."""
    texto = _fmt_ihr_bands()
    for cat, thr in IHR_THRESHOLDS.items():
        assert f"{thr:g}" in texto, f"Threshold {thr:g} ({cat}) no aparece en bandas: {texto}"


def test_bandas_citan_todos_los_labels():
    """Cada label de IHR_LABELS aparece en el texto."""
    texto = _fmt_ihr_bands()
    for cat, label in IHR_LABELS.items():
        assert label in texto, f"Label {label!r} ({cat}) no aparece en bandas: {texto}"


def test_bandas_orden_ascendente():
    """Los thresholds aparecen en orden ascendente en el texto."""
    texto = _fmt_ihr_bands()
    vals = sorted(IHR_THRESHOLDS.values())
    posiciones = [texto.find(f"{v:g}") for v in vals]
    assert posiciones == sorted(posiciones), f"Thresholds desordenados: {posiciones}"


def test_bandas_contiene_especulacion_extrema_fallback():
    """La categoria fallback (else de classify_ihr) aparece en las bandas."""
    texto = _fmt_ihr_bands()
    assert IHR_LABELS["especulacion_extrema"] in texto
    assert "<0.8" in texto or "<0.8 " in texto


def test_bandas_no_es_literal_hardcoded():
    """Source-level: sentiment.py usa el helper, no literal de 3 bandas."""
    import pathlib
    src = pathlib.Path("src/report/sentiment.py").read_text(encoding="utf-8")
    assert "_fmt_ihr_bands()" in src, "sentiment.py no invoca el helper"
    assert "bandas: <1.2 Especulacion, 1.2-1.6 Equilibrado" not in src, "literal viejo todavia presente"
