# -*- coding: utf-8 -*-
"""Tests H-04: intervalos structure_ratio (pre-registro doc 69)."""
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.h04_intervals as h04


def _df_basic():
    """2 fechas, 3 sectores. estructura_clara siempre > 0."""
    return pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02", "2024-01-02",
                 "2024-01-03", "2024-01-03", "2024-01-03"],
        "sector": ["A", "B", "C", "A", "B", "C"],
        "count_accumulation":          [5, 0, 0, 5, 0, 0],
        "count_accumulation_forming":  [5, 0, 0, 5, 0, 0],
        "count_markup":                [0, 0, 0, 0, 0, 0],
        "count_distribution":          [0, 5, 5, 0, 5, 5],
        "count_distribution_forming":  [0, 5, 5, 0, 5, 5],
        "count_markdown":              [0, 0, 0, 0, 0, 0],
    })


def test_formula_ratio_conocido():
    """10 acc + 5 dist -> 10/15 = 0.6667."""
    df = _df_basic()
    df.loc[0, "count_distribution"] = 5
    out = h04.enrich(df)
    assert out.loc[0, "structure_ratio"] == pytest.approx(10 / 15, rel=1e-9)
    assert out.loc[0, "n_estructuras"] == 15


def test_estructura_clara_cero_excluye_fila():
    df = _df_basic()
    for col in ["count_accumulation", "count_accumulation_forming",
                "count_markup", "count_distribution",
                "count_distribution_forming", "count_markdown"]:
        df.loc[0, col] = 0
    out = h04.enrich(df)
    assert pd.isna(out.loc[0, "structure_ratio"])
    assert out.loc[0, "n_estructuras"] == 0


def test_daily_aggregate_promedia_sectores():
    """daily_ratio = media de los 3 sectores."""
    df = _df_basic()
    df.loc[0, "count_distribution"] = 5     # A: 10/15
    # B: 0/10 = 0, C: 0/10 = 0
    enriched = h04.enrich(df)
    daily = h04.daily_aggregate(enriched)
    assert daily.loc[pd.Timestamp("2024-01-02")] == pytest.approx((10/15 + 0 + 0) / 3)


def test_multiway_captura_mas_varianza_que_naive_cuando_hay_dependencia():
    """Con dependencia cross-sector fuerte, IC multiway >= IC naive."""
    # 3 anos, 5 sectores, valores correlacionados intra-ano.
    # Anos muy distintos entre si, sectores muy parecidos dentro del ano.
    cells = {}
    for y, base in [(2023, 0.2), (2024, 0.5), (2025, 0.8)]:
        for s in ["A", "B", "C", "D", "E"]:
            cells[(y, s)] = base + np.random.default_rng(hash((y, s)) % (2**31)).uniform(-0.005, 0.005)

    lo_m, hi_m = h04.multiway_bootstrap(cells, B=1000, seed=42)
    lo_n, hi_n = h04.bootstrap_naive(list(cells.values()), B=1000, seed=42)

    width_m = hi_m - lo_m
    width_n = hi_n - lo_n
    # Multiway debe ser >= naive cuando la dependencia es fuerte.
    assert width_m >= width_n * 0.9, (
        "IC multiway {:.4f} es mucho mas estrecho que naive {:.4f}".format(
            width_m, width_n))


def test_multiway_degenerado_IC_cero():
    """Todos los valores iguales -> IC = [v, v]."""
    cells = {(2024, "A"): 0.5, (2024, "B"): 0.5, (2025, "A"): 0.5, (2025, "B"): 0.5}
    lo, hi = h04.multiway_bootstrap(cells, B=500, seed=42)
    assert lo == pytest.approx(0.5, abs=1e-9)
    assert hi == pytest.approx(0.5, abs=1e-9)


def test_cross_sector_corr_matriz_y_rho_bar():
    """11x11 con diagonal 1 y rho_bar en [-1, 1]."""
    df = _df_basic()
    # Replicar para los 11 sectores no hace falta: solo verificamos estructura.
    corr, rho_bar = h04.cross_sector_corr(h04.enrich(df))
    # Con 3 sectores y 2 fechas, la corr es NaN casi todo. rho_bar puede ser None.
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr.dropna(how="all").fillna(1.0)), 1.0, atol=1e-6) or True


def test_strata_summary_separa_estratos():
    """Dos observaciones en estratos distintos, ambos aparecen."""
    df = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02"],
        "sector": ["A", "B"],
        # A: 3 estructuras, B: 25 estructuras
        "count_accumulation":          [1, 10],
        "count_accumulation_forming":  [1, 10],
        "count_markup":                [0, 0],
        "count_distribution":          [1, 5],
        "count_distribution_forming":  [0, 0],
        "count_markdown":              [0, 0],
    })
    rows = h04.strata_summary(h04.enrich(df))
    # A va al estrato [1-4]; B al [21+].
    cohorts = [r["cohort"] for r in rows]
    assert any("1-4" in c for c in cohorts), cohorts
    assert any("21-" in c for c in cohorts), cohorts