# -*- coding: utf-8 -*-
"""FIX-FLOW-OTROS-N/D (2026-11-14): nota auto-generada de causa N/D."""
from src.report.leaders import render_momentum_otros


def test_sin_nan_no_anade_nota():
    out = render_momentum_otros(
        otros_price_rank=[("^GSPC", 0.01), ("^RUT", 0.02)],
        otros_flow_rank=[("^GSPC", 0.5), ("^RUT", 0.3)],
    )
    body = "".join(out)
    assert "N/D = z-score no computable" not in body


def test_un_nan_anade_nota_con_ticker():
    out = render_momentum_otros(
        otros_price_rank=[("^GSPC", 0.01), ("EURUSD=X", 0.001)],
        otros_flow_rank=[("^GSPC", 0.5), ("EURUSD=X", float("nan"))],
    )
    body = "".join(out)
    assert "N/D = z-score no computable" in body
    assert "EURUSD=X" in body
    nota = body.split("N/D = z-score no computable")[1]
    assert "^GSPC" not in nota


def test_varios_nan_lista_todos():
    out = render_momentum_otros(
        otros_price_rank=[("^GSPC", 0.01)],
        otros_flow_rank=[
            ("^GSPC", 0.5),
            ("EURUSD=X", float("nan")),
            ("USDJPY=X", float("nan")),
            ("^SPGSCI", float("nan")),
        ],
    )
    body = "".join(out)
    assert "EURUSD=X" in body
    assert "USDJPY=X" in body
    assert "^SPGSCI" in body
    assert "REP-AUDIT-20261017-H4" in body


def test_otros_flow_vacio_no_anade_nota():
    out = render_momentum_otros(
        otros_price_rank=[("^GSPC", 0.01)],
        otros_flow_rank=[],
    )
    body = "".join(out)
    assert "No hay datos disponibles" in body
    assert "N/D = z-score no computable" not in body
