# -*- coding: utf-8 -*-
"""REP-AUDIT-20261017-capa2b (H17): umbral de materialidad en la
etiqueta de divergencia precio-flujo.

Motivacion: el reporte publicaba 'Confirmacion de debilidad' para XLRE
(ret=-0.04%) y 'Absorcion potencial' para XLC (ret=-0.11%). Retornos
marginales no deben publicarse con etiquetas categoricas sin cualificar.

Ademas: el fichero contenia mojibake (U+00BE en lugar de U+00F3) en las
etiquetas. Verificado en auditoria del reporte 2026-10-17.
"""

from config.settings import FLOW_DIVERGENCE_MATERIALITY_THRESHOLD as UMBRAL
from indicators.sector_flow_characteristics import _regime


def test_h17_umbral_desde_config():
    assert 0 < UMBRAL < 0.05, (
        "Umbral de materialidad fuera de rango razonable: {!r}".format(UMBRAL)
    )


def test_h17_caso_xlre_ret_debil():
    """XLRE del reporte: ret=-0.04%, flujo negativo -> debilidad (debil)."""
    out = _regime(-0.0004, -1e8)
    assert out == "Confirmación de debilidad (débil)", "XLRE: got {!r}".format(out)


def test_h17_caso_xlc_ret_debil():
    """XLC del reporte: ret=-0.11%, flujo positivo -> absorcion (debil)."""
    out = _regime(-0.0011, 1e8)
    assert out == "Absorción potencial (débil)", "XLC: got {!r}".format(out)


def test_h17_caso_xlb_ret_material():
    """XLB del reporte: ret=+2.33%, flujo negativo -> divergencia SIN sufijo."""
    out = _regime(0.0233, -1e8)
    assert out == "Divergencia bajista", "XLB: got {!r}".format(out)
    assert "(débil)" not in out


def test_h17_etiquetas_fuertes_sin_mojibake():
    """Las 4 etiquetas fuertes usan UTF-8 correcto (tildes), no U+00BE."""
    casos = [
        (0.05, 1e8, "Confirmación"),
        (0.05, -1e8, "Divergencia bajista"),
        (-0.05, 1e8, "Absorción potencial"),
        (-0.05, -1e8, "Confirmación de debilidad"),
    ]
    for ret, flow, esperado in casos:
        out = _regime(ret, flow)
        assert out == esperado, (
            "ret={}, flow={}: esperado {!r}, got {!r}".format(
                ret, flow, esperado, out,
            )
        )
        # No debe contener el caracter mojibake
        assert "\u00be" not in out


def test_h17_ret_cero_neutral():
    """price_ret exactamente 0 -> Neutral (no debilidad ni confirmacion)."""
    assert _regime(0.0, 1e8) == "Neutral / sin confirmación"
    assert _regime(0.0, -1e8) == "Neutral / sin confirmación"


def test_h17_flow_cero_neutral():
    """flow_sum exactamente 0 -> Neutral (no hay direccion de flujo)."""
    assert _regime(0.05, 0) == "Neutral / sin confirmación"
    assert _regime(-0.05, 0) == "Neutral / sin confirmación"


def test_h17_nan_ambos_devuelve_nd():
    """NaN en cualquiera de los dos -> N/D."""
    assert _regime(float("nan"), 1e8) == "N/D"
    assert _regime(0.05, float("nan")) == "N/D"
    assert _regime(float("nan"), float("nan")) == "N/D"
