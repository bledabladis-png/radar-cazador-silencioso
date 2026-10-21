# -*- coding: utf-8 -*-
"""Tests de notas derivadas de constantes (2026-10-21).

Las notas del reporte deben derivar sus numeros de las constantes
en runtime, no hardcodearlos. Si cambia STRUCTURAL_WEIGHTS,
FLOW_PROXY_WEIGHTS o EXPECTED_SECTOR_COUNT, la nota debe reflejar
el nuevo valor sin tocar el codigo.

3 tests, uno por nota:
- H1: nota Structural vs Struct Wyckoff -> STRUCTURAL_WEIGHTS + STRUCT_STRONG
- P1: nota Flow Proxy -> FLOW_PROXY_WEIGHTS
- P2c: nota regla Wyckoff -> EXPECTED_SECTOR_COUNT
"""
import pandas as pd

import src.report.rankings as rankings_mod
import src.report.leaders as leaders_mod
import src.report.synthesis as synthesis_mod


def _make_sector_results():
    return {
        "ranking": [("XLP", "Staples", 0.6, "RANGE")],
        "struct_scores": {"XLP": 0.17},
        "scores": None,
        "regime": "ROT",
        "last_scores": None,
    }


def _render_rankings():
    return "".join(rankings_mod.render_ranking_sectorial_unificado(
        _make_sector_results(),
        tactical_scores=None,
        structural_scores=None,
        sector_persistence=None,
        signal_agreements=None,
        signal_agreements_display=None,
        shock_sensitivities=None,
    ))


def _render_momentum():
    return "".join(leaders_mod.render_momentum_sectores(
        sector_price_rank=[("XLK", 0.01)],
        sector_flow_rank=[("XLK", 0.5)],
    ))


def _make_evidence_df():
    return pd.DataFrame([{
        "sector": "XLE",
        "price_evidence": 1, "breadth_evidence": 1,
        "primary_flow_evidence": 1, "proxy_flow_evidence": 1,
        "wyckoff_evidence": 1, "credit_evidence": 0,
        "volatility_evidence": 0,
        "evidence_quality": "Alta",
        "alignment_reading": "EVIDENCIA PREDOMINANTEMENTE FAVORABLE",
    }])


def _render_matriz():
    return "".join(synthesis_mod.render_matriz_evidencia(_make_evidence_df()))


# ---------- H1: nota deriva de STRUCTURAL_WEIGHTS y STRUCT_STRONG ----------

def test_h1_nota_deriva_de_structural_weights(monkeypatch):
    """Si cambian los pesos de STRUCTURAL_WEIGHTS, la nota los refleja."""
    monkeypatch.setitem(
        rankings_mod.STRUCTURAL_WEIGHTS, "rs_structural", 0.42
    )
    monkeypatch.setitem(
        rankings_mod.STRUCTURAL_WEIGHTS, "flow_structure", 0.33
    )
    monkeypatch.setitem(
        rankings_mod.STRUCTURAL_WEIGHTS, "persistence", 0.25
    )
    body = _render_rankings()
    assert "0.42*rs_structural" in body
    assert "0.33*flow_structure" in body
    assert "0.25*persistence" in body


def test_h1_nota_deriva_de_struct_strong(monkeypatch):
    """Si cambia STRUCT_STRONG, la nota lo refleja."""
    monkeypatch.setattr(rankings_mod, "STRUCT_STRONG", 0.45)
    body = _render_rankings()
    assert "STRUCT_STRONG=0.45" in body


# ---------- P1: nota deriva de FLOW_PROXY_WEIGHTS ----------

def test_p1_nota_deriva_de_flow_proxy_weights(monkeypatch):
    """Si cambian los pesos de FLOW_PROXY_WEIGHTS, la nota los refleja."""
    monkeypatch.setitem(leaders_mod.FLOW_PROXY_WEIGHTS, "flow_smooth", 0.20)
    monkeypatch.setitem(leaders_mod.FLOW_PROXY_WEIGHTS, "obv", 0.40)
    monkeypatch.setitem(leaders_mod.FLOW_PROXY_WEIGHTS, "cmf", 0.40)
    body = _render_momentum()
    assert "0.20*flow_smooth" in body
    assert "0.40*obv_z" in body
    assert "0.40*cmf_z" in body


# ---------- P2c: nota deriva de EXPECTED_SECTOR_COUNT ----------

def test_p2c_nota_deriva_de_expected_sector_count(monkeypatch):
    """Si cambia EXPECTED_SECTOR_COUNT, la nota lo refleja."""
    monkeypatch.setattr(synthesis_mod, "EXPECTED_SECTOR_COUNT", 15)
    body = _render_matriz()
    assert "15 sectores pueden coincidir" in body
