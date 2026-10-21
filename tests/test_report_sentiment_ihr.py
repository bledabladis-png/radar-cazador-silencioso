# -*- coding: utf-8 -*-
"""P4 (2026-10-21, auditor externo): el IHR es un proxy de sesgo
de cobertura, no una medida de participacion institucional.
"""
from src.report.sentiment import render_sentimiento_opciones


def _pcr_data(ihr=1.80):
    return {
        "total_pcr": 0.87,
        "index_pcr": 1.01,
        "equity_pcr": 0.56,
        "etp_pcr": 0.94,
        "vix_pcr": 0.53,
        "spx_pcr": 1.15,
        "ihr": ihr,
        "ihr_state": "Cobertura institucional alta",
        "index_volume_share": 0.553,
        "put_share": 0.466,
        "call_share": 0.534,
        "volume_pcr": 0.87,
        "oi_pcr": 0.75,
        "last_date": "2026-10-20",
        "timestamp": "2026-10-21 01:46:47",
    }


def test_p4_label_renombrado():
    """El label ya no dice 'Institutional Hedge Ratio'."""
    out = "".join(render_sentimiento_opciones(_pcr_data()))
    assert "Institutional Hedge Ratio" not in out
    assert "Index/Equity PCR Ratio" in out
    assert "proxy" in out.lower()


def test_p4_formula_declarada():
    """La nota declara la formula del ratio."""
    out = "".join(render_sentimiento_opciones(_pcr_data()))
    assert "index_pcr / equity_pcr" in out


def test_p4_nota_no_identifica_institucionales():
    """La nota aclara que no identifica participantes."""
    out = "".join(render_sentimiento_opciones(_pcr_data()))
    assert "no identifican participantes institucionales" in out
    assert "agregados" in out


def test_p4_valor_sigue_visible():
    """El valor 1.80 sigue publicado."""
    out = "".join(render_sentimiento_opciones(_pcr_data(ihr=1.80)))
    assert "1.80" in out
