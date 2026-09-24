# -*- coding: utf-8 -*-
"""Tests de regresion para el fix C2 (render_representatividad_lider).

Contexto: el render iteraba el DataFrame completo sin filtrar a la ultima
fecha. Si el df traia N fechas (historico), concatenaba bloques sin
etiqueta. El sintoma observado en el reporte: mismo ticker repetido 2-3
veces por sector.
"""

import pandas as pd

from src.report.sector_context import render_representatividad_lider


def _make_df(dates, tickers_per_date):
    rows = []
    for d in dates:
        for sector, ticker in tickers_per_date:
            rows.append({
                'date': pd.Timestamp(d),
                'sector': sector,
                'ticker': ticker,
                'rs_distance_to_median': 0.0,
                'mom_distance_to_median': 0.0,
                'flow_distance_to_median': 0.0,
                'wls_distance_to_median': 0.0,
                'sector_rank_pct': 0.5,
            })
    return pd.DataFrame(rows)


def test_c2_solo_ultima_fecha():
    """Con 2 fechas, solo se renderiza la ultima."""
    df = _make_df(
        ["2026-09-17", "2026-09-18"],
        [("XLF", "JPM"), ("XLF", "WFC")],
    )
    out = render_representatividad_lider(df)
    body = "".join(out)
    assert body.count("JPM") == 1
    assert body.count("WFC") == 1


def test_c2_no_muta_input():
    """El fix copia el df; no debe mutar el original."""
    df = _make_df(["2026-09-17", "2026-09-18"], [("XLF", "JPM")])
    n_before = len(df)
    _ = render_representatividad_lider(df)
    assert len(df) == n_before


def test_c2_df_vacio():
    """df vacio -> no imprime nada."""
    out = render_representatividad_lider(pd.DataFrame())
    assert out == []


def test_c2_sin_columna_date():
    """Sin columna date, se comporta como antes (no filtra)."""
    df = pd.DataFrame([{
        'sector': 'XLF', 'ticker': 'JPM',
        'rs_distance_to_median': 0.0,
        'mom_distance_to_median': 0.0,
        'flow_distance_to_median': 0.0,
        'wls_distance_to_median': 0.0,
        'sector_rank_pct': 0.5,
    }])
    out = render_representatividad_lider(df)
    assert any("JPM" in line for line in out)