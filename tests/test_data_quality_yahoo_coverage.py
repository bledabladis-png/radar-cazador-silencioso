# -*- coding: utf-8 -*-
"""Regresion H1 (2026-10-17): Yahoo Finance debe reportar cobertura.

El commit 1a4e7ea2 (2026-09-12) cambio `Yahoo Finance` de
`sector_breadth.csv` (coverage_logic='sectorial') a
`market_data.parquet` (coverage_logic=None). La rama parquet escribia
coverage=NaN. Desde ese dia el reporte mostraba N/D de cobertura para
la fuente principal de precios.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators.data_quality import compute_data_quality


def test_yahoo_finance_tiene_coverage_sectorial():
    """Yahoo Finance reporta n_total=11, coverage calculable."""
    import os
    os.chdir(str(ROOT))
    df = compute_data_quality()
    yahoo = df[df["source"] == "Yahoo Finance"].iloc[-1]
    assert yahoo["n_total"] == 11
    assert pd.notna(yahoo["n_valid"])
    assert pd.notna(yahoo["coverage"])
    assert 0.0 <= yahoo["coverage"] <= 1.0