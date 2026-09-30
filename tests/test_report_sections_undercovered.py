"""Tests report: confirmation + darkpool + sentiment + volatility_mte.

4 modulos de presentacion pura (render_* -> list[str]). Verifican:
  - Casos borde (None, dict vacio, df vacio).
  - Ramas de render (con datos -> lineas; sin datos -> lista vacia).
  - Manejo de NaN/N-D.
  - Contratos de formato especificos de cada seccion.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.report.confirmation import render_confirmation
from src.report.darkpool import render_darkpool
from src.report.sentiment import render_sentimiento_opciones
from src.report.volatility_mte import (
    render_estructura_volatilidad, render_calidad_datos, render_mte,
)


# =============================================================================
# confirmation.py
# =============================================================================

def test_confirmation_none_devuelve_vacio():
    assert render_confirmation(None) == []


def test_confirmation_dict_vacio_devuelve_vacio():
    assert render_confirmation({}) == []


def test_confirmation_t10y3m_positivo_lleva_signo():
    out = render_confirmation({"t10y3m": 1.23})
    lineas = "".join(out)
    assert "+1.23%" in lineas


def test_confirmation_t10y3m_negativo_sin_signo_extra():
    out = render_confirmation({"t10y3m": -0.45})
    lineas = "".join(out)
    assert "-0.45%" in lineas
    assert "**10Y-3M Spread:**" in lineas


def test_confirmation_nan_renderiza_nd():
    out = render_confirmation({"rv_21d": np.nan, "rv_60d": np.nan})
    lineas = "".join(out)
    assert lineas.count("N/D") >= 2


def test_confirmation_fls_con_desglose():
    out = render_confirmation({
        "fls": {
            "fls_normalized": 0.69,
            "components": 3,
            "total_components": 5,
            "stressed_components": 3,
            "detail": {
                "SOFR": {"stressed": True, "value": 1.0},
                "RRP": {"stressed": False, "value": -1.0},
            },
        },
    })
    lineas = "".join(out)
    assert "Funding & Liquidity Stress (FLS)" in lineas
    assert "WARN SOFR" in lineas
    assert "OK RRP" in lineas
    assert "3/5 componentes" in lineas


def test_confirmation_ad_universo_completo():
    out = render_confirmation({
        "ad": {"ad_net": 74, "advances": 193, "declines": 119,
               "new_highs": 5, "new_lows": 2, "nh_nl": 3,
               "breadth_thrust": 0.5, "ad_line": 14843},
    })
    lineas = "".join(out)
    assert "Advance/Decline Net (universo completo)" in lineas
    assert "universo completo" in lineas


def test_confirmation_breadth_thrust_extremo():
    out = render_confirmation({"ad": {"breadth_thrust": 0.85, "ad_line": 0}})
    lineas = "".join(out)
    assert "Breadth Thrust extremo" in lineas


def test_confirmation_breadth_thrust_normal_no_aparece():
    out = render_confirmation({"ad": {"breadth_thrust": 0.5, "ad_line": 0}})
    lineas = "".join(out)
    assert "Breadth Thrust extremo" not in lineas


def test_confirmation_recession_capitulation_signal():
    out = render_confirmation({
        "ad": {"nh_nl": -5, "ad_line": 0},
        "mte_scenario": "RECESSION",
    })
    lineas = "".join(out)
    assert "RECESSION CAPITULATION SIGNAL" in lineas


def test_confirmation_ratios_tabla():
    out = render_confirmation({
        "ratios": {
            "copper_gold": 0.0123,
            "copper_gold_delta20": 0.05,
            "copper_gold_zscore": 0.62,
            "hyg_lqd": 0.7544,
            "hyg_lqd_delta20": None,
            "hyg_lqd_zscore": 0.49,
        },
    })
    lineas = "".join(out)
    assert "### Cross-Asset Ratios" in lineas
    assert "Copper/Gold" in lineas
    assert "0.0123" in lineas
    assert "+5.0%" in lineas
    assert "+0.62" in lineas
    # HYG/LQD con delta None -> N/D
    assert "N/D" in lineas


# =============================================================================
# darkpool.py
# =============================================================================

def test_darkpool_none_devuelve_vacio():
    assert render_darkpool(None) == []


def test_darkpool_dict_vacio_devuelve_vacio():
    assert render_darkpool({}) == []


def test_darkpool_basico():
    out = render_darkpool({
        "media_dark_pool": 22.66,
        "n_tickers_ats": 536,
        "n_tickers_total": 536,
        "week": "N/A",
    })
    lineas = "".join(out)
    assert "Actividad en ATS" in lineas
    assert "22.66%" in lineas
    assert "536/536" in lineas
    assert "Semana FINRA:** N/D" in lineas


def test_darkpool_z_windows():
    out = render_darkpool({
        "media_dark_pool": 22.0, "n_tickers_ats": 100, "n_tickers_total": 100,
        "week": "N/A",
        "z_windows": {
            "13w": {"z": 0.31, "state": "Actividad ATS normal"},
            "52w": {"z": -0.69, "state": "Actividad ATS baja"},
        },
    })
    lineas = "".join(out)
    assert "Z-Scores por ventana" in lineas
    assert "13w: Z=0.31" in lineas
    assert "52w: Z=-0.69" in lineas


def test_darkpool_fallback_z_score_sin_windows():
    out = render_darkpool({
        "media_dark_pool": 22.0, "n_tickers_ats": 100, "n_tickers_total": 100,
        "week": "N/A",
        "z_score": 0.31, "momentum": -0.1, "percentile": 40, "state": "NORMAL",
    })
    lineas = "".join(out)
    assert "Robust Z-Score:** 0.31" in lineas
    assert "Momentum:** -0.10" in lineas


def test_darkpool_sin_z_acumulando_historial():
    out = render_darkpool({
        "media_dark_pool": 22.0, "n_tickers_ats": 100, "n_tickers_total": 100,
        "week": "N/A",
        "z_score": np.nan,
    })
    lineas = "".join(out)
    assert "Acumulando historial" in lineas


def test_darkpool_top5_tabla():
    df = pd.DataFrame({
        "ticker": ["Q", "RL", "EFX", "CRL", "DHR", "SPY"],
        "dark_pool_pct": [35.85, 35.71, 35.67, 35.39, 34.86, 10.0],
        "ats_volume": [3_473_222, 1_374_086, 2_062_112, 1_338_959, 5_239_566, 1_000],
        "total_volume": [9_689_100, 3_848_100, 5_780_800, 3_783_600, 15_030_900, 10_000],
    })
    out = render_darkpool({
        "media_dark_pool": 22.0, "n_tickers_ats": 100, "n_tickers_total": 100,
        "week": "N/A",
        "datos": df,
    })
    lineas = "".join(out)
    assert "Mayor % de volumen en ATS" in lineas
    assert "| Q |" in lineas
    # SPY con 10% no aparece en top-5
    assert "| SPY |" not in lineas


def test_darkpool_freshness_archival_alerta():
    old_week = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    out = render_darkpool({
        "media_dark_pool": 22.0, "n_tickers_ats": 100, "n_tickers_total": 100,
        "week": old_week,
    })
    lineas = "".join(out)
    assert "DATOS OBSOLETOS" in lineas


# =============================================================================
# sentiment.py
# =============================================================================

def test_sentiment_none_devuelve_vacio():
    assert render_sentimiento_opciones(None) == []


def test_sentiment_dict_vacio_devuelve_vacio():
    assert render_sentimiento_opciones({}) == []


def test_sentiment_basico():
    out = render_sentimiento_opciones({
        "total_pcr": 0.75, "pcr_ewm": 0.80,
        "z_score": -1.31, "momentum": -0.41, "percentile": 2,
        "state": "Optimismo",
        "index_pcr": 0.95, "equity_pcr": 0.52, "etp_pcr": 0.77,
        "vix_pcr": 0.29, "spx_pcr": 1.06,
        "ihr": 1.83, "ihr_state": "Cobertura institucional alta",
        "index_volume_share": 0.445,
        "put_share": 0.428, "call_share": 0.572,
        "volume_pcr": 0.75, "oi_pcr": 0.75,
        "last_date": "2026-09-25",
        "timestamp": "2026-09-26 14:12:53",
    })
    lineas = "".join(out)
    assert "## Sentimiento de Opciones" in lineas
    assert "PCR Total:** 0.75" in lineas
    assert "EWMA(5): 0.80" in lineas
    assert "Optimismo" in lineas
    assert "44.5%" in lineas
    assert "42.8%" in lineas


def test_sentiment_ewma_nan_muestra_nd():
    out = render_sentimiento_opciones({
        "total_pcr": 0.75, "pcr_ewm": np.nan,
        "z_score": np.nan,
    })
    lineas = "".join(out)
    assert "EWMA(5): N/D - historial insuficiente" in lineas


def test_sentiment_sin_z_no_muestra_bloque_zscore():
    out = render_sentimiento_opciones({
        "total_pcr": 0.75, "pcr_ewm": 0.80, "z_score": np.nan,
    })
    lineas = "".join(out)
    assert "Robust Z-Score" not in lineas


# =============================================================================
# volatility_mte.py
# =============================================================================

def test_vol_structure_none_devuelve_vacio():
    assert render_estructura_volatilidad(None) == []


def test_vol_structure_df_vacio_devuelve_vacio():
    assert render_estructura_volatilidad(pd.DataFrame()) == []


def test_vol_structure_tabla():
    df = pd.DataFrame([{
        "date": pd.Timestamp("2026-09-25"),
        "vix_level": 14.87, "vix_percentile_20d": 0.28, "vix_percentile_60d": 0.17,
        "term_structure_ratio": 1.21, "pcr_zscore": -1.31, "pcr_percentile_20d": 0.0,
        "volatility_reading": "Volatilidad por debajo de la media",
        "term_structure_reading": "Contango (vol. implicita mayor horizonte superior a inmediata)",
    }])
    out = render_estructura_volatilidad(df)
    lineas = "".join(out)
    assert "## Estructura de volatilidad" in lineas
    assert "2026-09-25" in lineas
    assert "14.87" in lineas


def test_calidad_datos_none_devuelve_vacio():
    assert render_calidad_datos(None) == []


def test_calidad_datos_df_vacio_devuelve_vacio():
    assert render_calidad_datos(pd.DataFrame()) == []


def test_calidad_datos_una_fila_por_source():
    df = pd.DataFrame([
        {"date": "2026-09-23", "source": "Yahoo", "last_date": "2026-09-23",
         "age_calendar_days": 3, "frequency": "daily", "freshness": "CURRENT",
         "coverage": 1.0, "notes": ""},
        {"date": "2026-09-25", "source": "Yahoo", "last_date": "2026-09-25",
         "age_calendar_days": 1, "frequency": "daily", "freshness": "CURRENT",
         "coverage": 1.0, "notes": ""},
        {"date": "2026-09-25", "source": "FINRA", "last_date": "2026-08-31",
         "age_calendar_days": 26, "frequency": "finra", "freshness": "CURRENT",
         "coverage": np.nan, "notes": "Retraso regulatorio"},
    ])
    out = render_calidad_datos(df)
    lineas = "".join(out)
    # Debe mostrar solo la fila más reciente de cada source
    assert lineas.count("| Yahoo |") == 1
    assert "2026-09-25" in lineas
    assert "N/D" in lineas  # coverage NaN en FINRA


def test_mte_none_devuelve_vacio():
    assert render_mte(None) == []


def test_mte_dict_vacio_devuelve_vacio():
    assert render_mte({}) == []


def test_mte_unconfirmed_si_confidence_baja():
    out = render_mte({
        "scenario": "MIXED", "confidence": 0.30,
        "msi": 28, "ipi": 49, "srs": 0.01, "shs": -0.57, "cls": 0.07, "ips": -0.02,
    })
    lineas = "".join(out)
    assert "Escenario (UNCONFIRMED):" in lineas
    assert "No se considera confirmado" in lineas


def test_mte_confirmado_si_confidence_alta():
    out = render_mte({
        "scenario": "CRISIS", "confidence": 0.75,
        "msi": 80, "ipi": 50, "srs": 0.5, "shs": 0.5, "cls": 0.9, "ips": 0.3,
    })
    lineas = "".join(out)
    assert "**Escenario:** CRISIS" in lineas
    assert "(UNCONFIRMED)" not in lineas


def test_mte_nan_en_scores():
    out = render_mte({
        "scenario": "MIXED", "confidence": np.nan,
        "msi": 28, "ipi": 49, "srs": np.nan, "shs": np.nan,
        "cls": np.nan, "ips": np.nan,
    })
    lineas = "".join(out)
    assert "N/D" in lineas


# ---------- D6b: propagacion de reference_date ----------
def test_render_darkpool_usa_reference_date():
    """D6b (2026-09-30): retraso calculado contra reference_date."""
    from datetime import datetime
    dp_data = {'week': '2026-09-15', 'media_dark_pool': 0,
               'n_tickers_ats': 0, 'n_tickers_total': 0}
    ref = datetime(2026, 9, 30, 10, 0, 0)
    out = render_darkpool(dp_data, reference_date=ref)
    joined = ''.join(out)
    assert 'retraso: 15 dias' in joined, joined


def test_render_sentimiento_usa_reference_date():
    """D6b (2026-09-30): desfase calculado contra reference_date."""
    from datetime import datetime
    pcr_data = {'last_date': '2026-09-20', 'total_pcr': 1.0}
    ref = datetime(2026, 9, 30, 10, 0, 0)
    out = render_sentimiento_opciones(pcr_data, reference_date=ref)
    joined = ''.join(out)
    assert '(desfase: 10 dias)' in joined, joined
