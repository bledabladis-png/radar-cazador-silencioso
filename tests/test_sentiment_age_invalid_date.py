# -*- coding: utf-8 -*-
"""Centinela H5.2: render_sentimiento_opciones con last_date invalido."""

from src.report.sentiment import render_sentimiento_opciones


def _base(last_date):
    return {
        "total_pcr": 0.87, "index_pcr": 1.01, "equity_pcr": 0.56,
        "etp_pcr": 0.94, "vix_pcr": 0.53, "spx_pcr": 1.15,
        "ihr": 1.80, "ihr_state": "Cobertura institucional alta",
        "index_volume_share": 0.553, "put_share": 0.466, "call_share": 0.534,
        "volume_pcr": 0.87, "oi_pcr": 0.75,
        "last_date": last_date, "timestamp": "2026-10-21 01:46:47",
    }


def test_last_date_valido_calcula_desfase():
    out = "".join(render_sentimiento_opciones(_base("2026-10-20")))
    assert "desfase:" in out
    assert "N/D - fecha no parseable" not in out


def test_last_date_invalido_muestra_nd_visible():
    out = "".join(render_sentimiento_opciones(_base("2026-XX-YY")))
    assert "N/D - fecha no parseable" in out
    assert "desfase: N/D" in out


def test_last_date_na_no_pinta_desfase():
    out = "".join(render_sentimiento_opciones(_base("N/A")))
    assert "desfase" not in out
