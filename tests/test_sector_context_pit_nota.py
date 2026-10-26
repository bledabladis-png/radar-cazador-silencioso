# -*- coding: utf-8 -*-
"""Tests de la nota P1-H04 en render_wyckoff_sectorial.

El dictamen 2026-10-26 (P1-H04) exige que la limitacion de composicion
historica (2026Q4 aplicada retrospectivamente) sea visible en los
artefactos consumidos, no solo en doc de auditoria.
"""
import pandas as pd

from src.report.sector_context import render_wyckoff_sectorial


def _df_ejemplo():
    return pd.DataFrame([{
        "sector": "XLI", "date": "2026-10-23",
        "coverage_wyckoff": 100.0,
        "pct_accumulation": 0, "pct_markup": 0, "pct_range": 100,
        "pct_distribution": 0, "pct_markdown": 0, "pct_conflict": 0,
        "pct_accumulation_forming": 0, "pct_distribution_forming": 0,
        "n_valid_wyckoff": 10,
    }])


def test_pit_nota_presente_con_df():
    text = "".join(render_wyckoff_sectorial(_df_ejemplo()))
    assert "Composicion de holdings" in text
    assert "retrospectivamente" in text
    assert "point-in-time" in text
    assert "2026Q4" in text


def test_pit_nota_presente_sin_df():
    # Incluso sin tabla, la nota se emite (coherente con H6).
    text = "".join(render_wyckoff_sectorial(None))
    assert "Composicion de holdings" in text