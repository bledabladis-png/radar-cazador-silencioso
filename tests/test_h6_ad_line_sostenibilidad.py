# -*- coding: utf-8 -*-
"""REP-AUDIT-20261017-capa5 (H6): la A/D Line declara que se recalcula
en cada run, no es serie incremental persistida.

Motivacion: el auditor (2026-10-17) detecto que la A/D Line paso de
15.010 a 17.455 entre runs consecutivos cuando el ad_net diario era
+75. El valor no se persiste: se recalcula sobre el universo actual
de df_stocks en cada run. Cambios en el universo o revisiones de
Yahoo pueden alterar el valor absoluto aunque el flujo diario sea
pequeno. El reporte debe declararlo.
"""
from src.report.confirmation import render_confirmation


def _run(ad_dict):
    return "".join(render_confirmation({"ad": ad_dict}))


def test_h6_render_muestra_valor_ad_line():
    out = _run({"ad_line": 17455, "advances": 188, "declines": 113})
    assert "A/D Line (acumulada):" in out
    assert "17455" in out


def test_h6_render_declara_recalculo():
    out = _run({"ad_line": 17455})
    assert "Recalculada sobre el universo actual" in out
    assert "no es una serie incremental persistida" in out
    assert "no es comparable con runs anteriores" in out


def test_h6_render_incluye_universo_si_disponible():
    out = _run({"ad_line": 17455, "active_tickers": 305, "total_tickers": 329})
    assert "Universo: 305/329 tickers" in out


def test_h6_render_sin_universo_no_falla():
    """Tests existentes pasan ad sin active_tickers: no debe romper.

    La declaracion se emite igualmente (sin numeros).
    """
    out = _run({"ad_line": 14843, "breadth_thrust": 0.5})
    assert "14843" in out
    assert "Recalculada sobre el universo actual" in out
    assert "Universo:" not in out


def test_h6_render_ad_line_cero():
    """ad_line=0 es valor legitimo (universo plano), no ausencia."""
    out = _run({"ad_line": 0})
    assert "A/D Line (acumulada):" in out
    assert "Recalculada sobre el universo actual" in out


def test_h6_ad_dict_vacio_no_emite_bloque():
    """ad = {} (falsy) salta el bloque completo, sin errores."""
    out = _run({})
    assert "A/D Line" not in out
