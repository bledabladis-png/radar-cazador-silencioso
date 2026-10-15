"""Test H4: el reporte aclara que "Volatilidad LOW" (velocidad) y
"Lectura volatilidad media" (nivel) miden cosas distintas.

Auditoria externa 2026-10-15. El auditor detecto que el reporte mostraba
"Volatility: LOW (Z=-0.55)" junto a "VIX Perc 20d=100%, Perc 60d=83%,
Volatilidad elevada reciente", aparentemente contradictorio. No lo es:
son dos metricas ortogonales.

Este test fija la nota aclaratoria en el render para que la confusion
no se reintroduzca.
"""
import pandas as pd

from src.report.header import render_regimenes
from src.report.volatility_mte import render_estructura_volatilidad


def test_header_nota_velocidad_vs_nivel():
    """El Resumen de Regimenes debe aclarar que Volatilidad es velocidad."""
    out = render_regimenes(
        macro_score=pd.Series([0.0]),
        macro_regime="MIXED",
        macro_conf=0.5,
        liquidity_score=pd.Series([0.0]),
        liquidity_regime="NEUTRAL",
        liq_conf=0.5,
        volatility_score=pd.Series([-0.77]),
        vol_regime="LOW",
        vol_conf=0.39,
        real_liquidity_regime="NEUTRAL",
        real_liquidity_conf=0.9,
        real_liq_score=pd.Series([0.0]),
        real_liq_prev=0.0,
        sector_regime="ROTATIONAL",
    )
    joined = "".join(out)
    assert "VELOCIDAD" in joined
    assert "NIVEL" in joined
    assert "z-score del retorno" in joined.lower() or "z-score del retorno" in joined
    assert "Estructura de volatilidad" in joined


def test_estructura_nota_nivel_vs_velocidad():
    """La tabla Estructura debe aclarar que mide el NIVEL del VIX."""
    df = pd.DataFrame([{
        "date": "2026-10-13",
        "vix_level": 15.27,
        "vix_percentile_20d": 0.50,
        "vix_percentile_60d": 0.42,
        "term_structure_ratio": 1.18,
        "term_structure_slope": 1.0,
        "pcr_zscore": 0.81,
        "pcr_percentile_20d": 0.70,
        "volatility_reading": "Volatilidad media",
        "term_structure_reading": "Contango",
    }])
    out = render_estructura_volatilidad(df)
    joined = "".join(out)
    assert "NIVEL" in joined
    assert "VELOCIDAD" in joined
    assert "Resumen de Regimenes" in joined
