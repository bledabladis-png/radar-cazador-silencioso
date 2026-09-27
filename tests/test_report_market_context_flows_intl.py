"""Tests report: market_context + flows_international.

Render puros. Verifican contratos, casos borde (None/df vacio) y formato.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.report.market_context import (
    render_liderazgo_interno, render_rotacion_reciente,
    render_dispersion_sectores, render_correlacion_sectores,
    render_contexto_cross_asset,
)
from src.report.flows_international import (
    render_flujo_daxex, render_flujo_isf, render_flujo_lyxi, render_flujo_iwm,
    render_flujo_qqq_sec, render_posicionamiento_cftc,
    render_flujo_posicional_nport, render_rendimiento_qqq,
    render_qqq_nport_flow, render_flujo_sintesis,
)


# =============================================================================
# market_context.py
# =============================================================================

def test_liderazgo_interno_none():
    assert render_liderazgo_interno(None) == []


def test_liderazgo_interno_df_vacio():
    assert render_liderazgo_interno(pd.DataFrame()) == []


def test_liderazgo_interno_top5_por_sector():
    df = pd.DataFrame([
        {"date": "2026-09-25", "sector": "XLK", "ticker": f"T{i}",
         "price_ret_20d": 0.5 - i*0.05, "rs_abs_20d": 0.1, "rs_internal_20d": 0.05,
         "classification": "Liderazgo relativo doble"}
        for i in range(8)
    ])
    out = render_liderazgo_interno(df)
    lineas = "".join(out)
    assert "## Liderazgo relativo interno" in lineas
    # Solo los top-5
    for i in range(5):
        assert f"T{i}" in lineas
    for i in range(5, 8):
        assert f"T{i}" not in lineas


def test_rotacion_reciente_none():
    assert render_rotacion_reciente(None) == []


def test_rotacion_reciente_basico():
    df = pd.DataFrame([{
        "sector": "XLK", "rank_actual": 1,
        "rank_change_5d": -2, "rank_change_10d": 0, "rank_change_20d": -6,
        "lectura_5d": "Estable", "lectura_10d": "Estable", "lectura_20d": "Fuerte mejora",
    }])
    out = render_rotacion_reciente(df)
    lineas = "".join(out)
    assert "Rotación sectorial reciente" in lineas
    assert "XLK" in lineas
    assert "-2" in lineas


def test_dispersion_none():
    assert render_dispersion_sectores(None) == []


def test_dispersion_basico():
    df = pd.DataFrame([{
        "date": pd.Timestamp("2026-09-25"),
        "range_pp": 12.68, "std_pp": 3.66, "mean_ret": -2.76,
        "dispersion_reading": "Baja", "heterogeneity_type": "Heterogeneidad amplia",
    }])
    out = render_dispersion_sectores(df)
    lineas = "".join(out)
    assert "2026-09-25" in lineas
    assert "12.68" in lineas


def test_correlacion_none():
    assert render_correlacion_sectores(None) == []


def test_correlacion_usa_ultima_fecha():
    df = pd.DataFrame([
        {"date": "2026-09-23", "window": 20, "corr_mean": 0.1, "corr_median": 0.1,
         "corr_p25": 0.0, "corr_p75": 0.2, "corr_min": -0.4, "corr_max": 0.7,
         "correlation_reading": "Co-movimiento bajo"},
        {"date": "2026-09-25", "window": 20, "corr_mean": 0.22, "corr_median": 0.31,
         "corr_p25": 0.03, "corr_p75": 0.44, "corr_min": -0.41, "corr_max": 0.69,
         "correlation_reading": "Co-movimiento bajo"},
    ])
    out = render_correlacion_sectores(df)
    lineas = "".join(out)
    # Solo la fila mas reciente
    assert "0.22" in lineas
    assert "0.1 " not in lineas and "0.10" not in lineas


def test_cross_asset_none():
    assert render_contexto_cross_asset(None) == []


def test_cross_asset_pivote_por_sector_window():
    df = pd.DataFrame([
        {"sector": "XLB", "window": 20, "asset_class": "equity", "mean_corr": 0.26},
        {"sector": "XLB", "window": 20, "asset_class": "rates", "mean_corr": 0.19},
        {"sector": "XLB", "window": 20, "asset_class": "credit", "mean_corr": 0.24},
        {"sector": "XLB", "window": 20, "asset_class": "commodities", "mean_corr": 0.12},
        {"sector": "XLB", "window": 20, "asset_class": "fx", "mean_corr": -0.17},
        {"sector": "XLB", "window": 20, "asset_class": "volatility", "mean_corr": -0.53},
    ])
    out = render_contexto_cross_asset(df)
    lineas = "".join(out)
    assert "Contexto transversal de mercado" in lineas
    assert "0.26" in lineas
    assert "-0.53" in lineas


# =============================================================================
# flows_international.py
# =============================================================================

def _daxex_df(**overrides):
    base = {
        "date": pd.Timestamp("2026-09-25"),
        "nav": 209.4063, "shares_outstanding": 40_331_816.0,
        "shares_change": -175_000.0,
        "estimated_flow_eur": -36_646_102.50,
        "flow_pct_assets": -0.0043,
        "flow_zscore": -5.00,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_daxex_none():
    assert render_flujo_daxex(None) == []


def test_daxex_basico():
    out = render_flujo_daxex(_daxex_df())
    lineas = "".join(out)
    assert "## Flujo Primario DAXEX" in lineas
    assert "209.4063" in lineas
    assert "-175,000" in lineas
    assert "-5.00" in lineas


def test_isf_none():
    assert render_flujo_isf(None) == []


def test_isf_basico():
    df = _daxex_df(nav=10.3712, shares_change=0.0,
                   estimated_flow_eur=0.0, flow_pct_assets=0.0,
                   flow_zscore=0.0)
    out = render_flujo_isf(df)
    lineas = "".join(out)
    assert "## Flujo Primario ISF.L" in lineas
    assert "10.3712" in lineas


def test_lyxi_none():
    assert render_flujo_lyxi(None) == []


def test_lyxi_con_shares_change_nd():
    """Si shares_change es NaN, se muestran lineas N/D."""
    df = pd.DataFrame([{
        "date": pd.Timestamp("2026-09-24"),
        "shares_outstanding": 2_131_521.0,
        "nav": 206.8488, "class_aum": 440_902_561.02,
        "shares_change": np.nan, "estimated_flow_eur": np.nan,
        "flow_pct_assets": np.nan, "flow_zscore": np.nan,
    }])
    out = render_flujo_lyxi(df)
    lineas = "".join(out)
    assert "N/D (histórico insuficiente)" in lineas
    assert "Flujo Estimado:** N/D" in lineas


def test_lyxi_con_shares_change_valido():
    df = pd.DataFrame([{
        "date": pd.Timestamp("2026-09-24"),
        "shares_outstanding": 2_131_521.0,
        "nav": 206.8488, "class_aum": 440_902_561.02,
        "shares_change": 5000.0, "estimated_flow_eur": 1_034_244.0,
        "flow_pct_assets": 0.0023, "flow_zscore": 0.11,
    }])
    out = render_flujo_lyxi(df)
    lineas = "".join(out)
    assert "5,000" in lineas
    assert "+0.11" in lineas


def test_iwm_none():
    assert render_flujo_iwm(None) == []


def test_iwm_basico():
    df = pd.DataFrame([{
        "date": pd.Timestamp("2026-09-25"),
        "nav": 281.7664, "shares_outstanding": 273_950_000.0,
        "shares_change": 1_300_000.0,
        "primary_flow_usd": 366_296_338.20,
        "primary_flow_pct": 0.0047,
        "primary_flow_z": 0.58,
    }])
    out = render_flujo_iwm(df)
    lineas = "".join(out)
    assert "## Flujo Primario IWM" in lineas
    assert "281.7664" in lineas


def test_qqq_sec_none():
    assert render_flujo_qqq_sec(None) == []


def test_qqq_sec_basico():
    df = pd.DataFrame([{
        "period_type": "semiannual", "period_end_date": "2026-03-31",
        "filing_date": "2026-06-01",
        "shares_sold": 445_700_000, "shares_repurchased": -444_650_000,
        "net_shares_flow": 1_050_000,
        "proceeds_shares_sold": 271_899_087_413.0,
        "value_shares_repurchased": -270_834_724_252.0,
        "primary_flow_usd": 1_064_363_161.0,
    }])
    out = render_flujo_qqq_sec(df)
    lineas = "".join(out)
    assert "SEMIANNUAL 2026-03-31" in lineas
    assert "445,700,000" in lineas


def test_cftc_none():
    assert render_posicionamiento_cftc(None) == []


def test_cftc_basico():
    df = pd.DataFrame([
        {"date": pd.Timestamp("2026-09-22"),
         "contract": "VIX FUTURES - CBOE", "participant": "dealer",
         "net_position": 66_943, "position_change": 4_142, "flow_z": 0.45},
    ])
    out = render_posicionamiento_cftc(df)
    lineas = "".join(out)
    assert "VIX FUTURES - CBOE" in lineas
    assert "dealer" in lineas


def test_nport_position_none_devuelve_placeholder():
    out = render_flujo_posicional_nport(None)
    lineas = "".join(out)
    assert "Sin datos N-PORT" in lineas


def test_nport_position_con_datos():
    df = pd.DataFrame([{
        "REPORT_DATE": pd.Timestamp("2026-03-31"),
        "REGISTRANT_NAME": "SELECT SECTOR SPDR",
        "ISSUER_NAME": "Norwegian Cruise",
        "IDENTIFIER_ISIN": "BMG667211046",
        "PREV_BALANCE": 2_543_505, "BALANCE": 2_429_969,
        "POSITION_CHANGE": -113_536, "POSITION_CHANGE_PCT": -4.4638,
    }])
    out = render_flujo_posicional_nport(df)
    lineas = "".join(out)
    assert "SELECT SECTOR SPDR" in lineas
    assert "-4.46%" in lineas


def test_rendimiento_qqq_none():
    assert render_rendimiento_qqq(None) == []


def test_rendimiento_qqq_con_effective_date():
    df = pd.DataFrame([{
        "displayLabel": "QQQ (Yahoo Finance)",
        "ytd": 21.61, "y1": 25.48, "y3": 107.71, "y5": 105.42,
        "y10": 578.92, "inception": 1632.11,
        "effectiveDate": "2026-09-25",
    }])
    out = render_rendimiento_qqq(df)
    lineas = "".join(out)
    assert "effectiveDate): 2026-09-25" in lineas
    assert "21.61%" in lineas


def test_rendimiento_qqq_fallback_as_of_date():
    """Sin effectiveDate, usa as_of_date truncado."""
    df = pd.DataFrame([{
        "displayLabel": "QQQ",
        "ytd": 1.0, "y1": 1.0, "y3": 1.0, "y5": 1.0, "y10": 1.0, "inception": 1.0,
        "as_of_date": "2026-09-26 14:20:55",
    }])
    out = render_rendimiento_qqq(df)
    lineas = "".join(out)
    assert "as_of_date): 2026-09-26" in lineas


def test_qqq_nport_flow_none_devuelve_placeholder():
    out = render_qqq_nport_flow(None)
    lineas = "".join(out)
    assert "Sin datos NPORT-P de QQQ" in lineas


def test_qqq_nport_flow_con_trimestre():
    df = pd.DataFrame([
        {"month": 1, "sales": 36_785_450_000, "redemptions": 37_516_130_000,
         "net_flow": -730_670_000, "report_date": "2026-03-31"},
    ])
    out = render_qqq_nport_flow(df)
    lineas = "".join(out)
    assert "Trimestre: Q1 2026" in lineas
    assert "36,785.45" in lineas


def test_flujo_sintesis_vacio():
    assert render_flujo_sintesis(None) == []
    assert render_flujo_sintesis({}) == []


def test_flujo_sintesis_basico():
    out = render_flujo_sintesis({
        "flow_proxy_sign": -0.21,
        "etf_primary_flow_sign": 0.12,
        "cftc_flow_sign": 0.11,
        "european_flow_sign": -0.88,
        "confidence": "MEDIA",
    })
    lineas = "".join(out)
    assert "FLOW_CONFIDENCE:** MEDIA" in lineas
    assert "-0.21" in lineas
    assert "-0.88" in lineas


def test_flujo_sintesis_european_default_cero():
    """Si european_flow_sign no esta, default es 0."""
    out = render_flujo_sintesis({"confidence": "BAJA"})
    lineas = "".join(out)
    assert "Europa Primary Flow | 0.00" in lineas