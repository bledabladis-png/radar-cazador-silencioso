# -*- coding: utf-8 -*-
"""Tests de regresion para el fix C3 (render_divergencia_sector_lideres).

Contexto: mismo patron que C2. El render iteraba el DataFrame completo
sin filtrar a la ultima fecha. Con historico de N fechas, los bloques
se concatenaban sin etiqueta.
"""

import pandas as pd

from src.report.sector_context import render_divergencia_sector_lideres


def _make_df(dates, sectors):
    rows = []
    for d in dates:
        for s in sectors:
            rows.append({
                'date': pd.Timestamp(d),
                'sector': s,
                'sector_ret_20d': 0.0,
                'n_leaders_positive': 3,
                'n_leaders_negative': 2,
                'n_leaders_beating_sector': 4,
                'n_leaders_valid': 5,
                'classification': 'Alineacion positiva',
            })
    return pd.DataFrame(rows)


def test_c3_solo_ultima_fecha():
    """Con 2 fechas, solo se renderiza la ultima."""
    df = _make_df(["2026-09-17", "2026-09-18"], ["XLF", "XLK"])
    out = render_divergencia_sector_lideres(df)
    body = "".join(out)
    assert body.count("XLF") == 1
    assert body.count("XLK") == 1


def test_c3_no_muta_input():
    """El fix copia el df; no debe mutar el original."""
    df = _make_df(["2026-09-17", "2026-09-18"], ["XLF"])
    n_before = len(df)
    _ = render_divergencia_sector_lideres(df)
    assert len(df) == n_before


def test_c3_df_vacio():
    out = render_divergencia_sector_lideres(pd.DataFrame())
    assert out == []


def test_c3_sin_columna_date():
    """Sin columna date, no filtra (comportamiento previo)."""
    df = pd.DataFrame([{
        'sector': 'XLF',
        'sector_ret_20d': 0.0,
        'n_leaders_positive': 3,
        'n_leaders_negative': 2,
        'n_leaders_beating_sector': 4,
        'n_leaders_valid': 5,
        'classification': 'Alineacion positiva',
    }])
    out = render_divergencia_sector_lideres(df)
    assert any("XLF" in line for line in out)