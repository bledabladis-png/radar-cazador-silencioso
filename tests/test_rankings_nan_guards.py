# -*- coding: utf-8 -*-
"""Centinela H6: rankings.py con NaN/None en persistencia y scores."""


from src.report.rankings import (
    render_ranking_sectorial_unificado,
    render_opportunity_map,
)


def _ranking(sector_results, **kwargs):
    return "".join(render_ranking_sectorial_unificado(sector_results, **kwargs))


def test_ranking_persistencia_nan_pinta_na():
    out = _ranking(
        {"ranking": [("XLK", "Tecnologia", 0.5)]},
        tactical_scores={"XLK": 0.1}, structural_scores={"XLK": 0.3},
        sector_persistence={"XLK": float("nan")},
        signal_agreements={"XLK": 0.6}, signal_agreements_display=None,
        shock_sensitivities=None,
    )
    fila = [l for l in out.split("\n") if "XLK" in l][0]
    assert "nan%" not in fila
    assert "N/A" in fila


def test_ranking_persistencia_none_no_mezcla_con_cero():
    out = _ranking(
        {"ranking": [("XLF", "Finanzas", 0.3)]},
        tactical_scores={"XLF": 0.2}, structural_scores={"XLF": 0.1},
        sector_persistence={"XLF": None},
        signal_agreements={"XLF": 0.5}, signal_agreements_display=None,
        shock_sensitivities=None,
    )
    fila = [l for l in out.split("\n") if "XLF" in l][0]
    assert "nan%" not in fila
    assert "N/A" in fila


def test_ranking_todo_presente_sigue_funcionando():
    out = _ranking(
        {"ranking": [("XLK", "Tecnologia", 0.5)]},
        tactical_scores={"XLK": 0.1}, structural_scores={"XLK": 0.3},
        sector_persistence={"XLK": 0.8},
        signal_agreements={"XLK": 0.6}, signal_agreements_display=None,
        shock_sensitivities=None,
    )
    fila = [l for l in out.split("\n") if "XLK" in l][0]
    assert "80%" in fila
    assert "70%" in fila


def test_opportunity_map_sin_nan_porcentaje():
    out = "".join(render_opportunity_map(
        tactical_scores={"XLK": float("nan"), "XLF": 0.1},
        structural_scores={"XLK": 0.3, "XLF": 0.1},
        sector_persistence={"XLK": None, "XLF": None},
        signal_agreements={"XLK": 0.6, "XLF": 0.5},
    ))
    assert "nan%" not in out
    assert "Signal Consistency" in out


def test_opportunity_map_medianas_con_notna():
    out = "".join(render_opportunity_map(
        tactical_scores={"XLK": float("nan")},
        structural_scores={"XLK": 0.3},
        sector_persistence={}, signal_agreements={},
    ))
    assert "Tactical mediana" in out
    assert "nan" not in out.split("\n")[2]
