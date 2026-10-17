# -*- coding: utf-8 -*-
"""Regresion H2 (2026-10-17): 12 fuentes con coverage_logic None.

Antes: 11 fuentes con `coverage_logic: None` en SOURCES -> el reporte
mostraba N/D en cobertura aunque el CSV tenia datos. Bug de
diagnostico, no de datos. El commit 1d3e0a17 (2026-10-10) declaro las
fuentes pero no implemento la logica.

Ahora: cada fuente tiene un coverage_logic especifico
('sectorial' / 'rows' / 'from_row') y declara los campos que necesita
(coverage_col o aliases n_total/n_sectors).

Excepciones documentadas:
- Leader Representativeness y Sector Leader Divergence: cobertura
  2/11 real (solo 2 sectores activos).
- Sector Correlation Summary: n_total=11 (sectores), n_valid=55
  (pares validos). El coverage real del CSV (0.9944) es el correcto.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators.data_quality import compute_data_quality

TARGETS = [
    "CFTC Position Flow", "FINRA Dark Pool", "Leader Representativeness",
    "Macro FRED", "RS Internal", "SEC N-PORT (posiciones)",
    "SEC N-PORT-P QQQ (flujos)", "Sector Correlation Summary",
    "Sector Dispersion", "Sector Leader Divergence",
    "Sector Rank History", "Volatility Structure",
]


def test_12_fuentes_coverage_no_nan():
    """Ninguna de las 12 fuentes con coverage_logic tiene NaN."""
    import os
    os.chdir(str(ROOT))
    df = compute_data_quality()
    missing = []
    for t in TARGETS:
        row = df[df["source"] == t].iloc[-1]
        if pd.isna(row["coverage"]):
            missing.append(t)
    assert not missing, (
        f"Fuentes con coverage NaN: {missing}. Todas deben declarar "
        "coverage_logic funcional."
    )