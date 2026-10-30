# -*- coding: utf-8 -*-
"""2026-10-30: semantica Fase dominante + n_valid explicito.

#2: columna "Fase Wyckoff" renombrada a "Fase dominante" en la Matriz
de Regimen Sectorial. La fase es la moda de la distribucion de fases
de holdings, no la fase del ETF (esa se retiro en sesion 3).

A: columna "N valid" en Concentracion del liderazgo. Expone cuantos
tickers del universo del ETF tienen datos suficientes para las
medianas. Antes solo se veia "Cob RS=18%" sin saber si eran 4 de 22
o 4 de 76.

B: columna "N valid NH/NL" en Sector Breadth & Health. Expone cuantos
componentes tienen >=253 sesiones de historial. Antes solo se veia
NH=0 / NL=0 sin saber si era "0 real" o "menos tickers".
"""
import pandas as pd

from src.report.sector_context import render_matriz_regimen
from src.report.sectorial import (
    render_sector_concentration,
    render_sector_breadth,
)


# ============================================================
# #2: header "Fase dominante" en Matriz de Regimen Sectorial
# ============================================================
def _matriz_df(n_valid=True):
    row = {
        "date": "2026-10-29",
        "sector": "XLK",
        "price_ret_20d": -0.045,
        "pct_above_ema50": 35.0,
        "flow_20d_sum": 1_758_001_577,
        "wyckoff_phase": "RANGE",
        "structure_ratio": 0.19,
        "n_estructuras": 16,
        "positive_conditions": 1,
        "regime_reading": "Mixto",
    }
    return pd.DataFrame([row])


def test_matriz_header_fase_dominante():
    out = "".join(render_matriz_regimen(_matriz_df()))
    assert "Fase dominante" in out
    assert "Fase Wyckoff" not in out


def test_matriz_nota_fase_dominante_aclaratoria():
    """La nota debe explicar que la fase es moda de holdings, no del ETF."""
    out = "".join(render_matriz_regimen(_matriz_df()))
    assert "NO es la fase del ETF sectorial" in out
    assert "moda de la distribucion de fases" in out


# ============================================================
# A: columna "N valid" en Concentracion del liderazgo
# ============================================================
def _conc_df(with_n_valid=True):
    row = {
        "date": "2026-10-29",
        "sector": "XLF",
        "top1_positive_return_concentration": 0.107,
        "top3_positive_return_concentration": 0.282,
        "top5_positive_return_concentration": 0.432,
        "rs_median": 4.86,
        "momentum_median": -0.0001,
        "flow_median": 0.66,
        "wyckoff_median": 0.10,
        "wls_median": 0.05,
        "leader_ticker": "SPGI",
        "leader_return20": 0.1868,
        "coverage_rs": 18.0,
        "coverage_momentum": 18.0,
        "coverage_flow": 18.0,
        "coverage_wyckoff": 18.0,
        "coverage_wls": 18.0,
    }
    if with_n_valid:
        row["n_valid_rs"] = 4
    return pd.DataFrame([row])


def test_concentracion_header_con_n_valid():
    out = "".join(render_sector_concentration(_conc_df()))
    assert "N valid" in out
    # Cabecera: 17 columnas (antes 16)
    header_line = [ln for ln in out.split("\n") if ln.startswith("| Sector |")][0]
    assert header_line.count("|") == 18  # 17 columnas -> 18 pipes


def test_concentracion_n_valid_valor():
    out = "".join(render_sector_concentration(_conc_df()))
    # El valor 4 debe aparecer antes de coverage 18%
    assert " | 4 | 18% | " in out


def test_concentracion_sin_columna_usa_nd():
    """df antiguo (sin n_valid_rs): muestra N/D, no rompe."""
    out = "".join(render_sector_concentration(_conc_df(with_n_valid=False)))
    assert " | N/D | 18% | " in out


def test_concentracion_nota_n_valid_aclaratoria():
    out = "".join(render_sector_concentration(_conc_df()))
    assert "N valid = numero de tickers" in out
    assert "interseccion comun" in out


# ============================================================
# B: columna "N valid NH/NL" en Sector Breadth & Health
# ============================================================
def _breadth_df(with_n_valid_nhnl=True):
    row = {
        "date": "2026-10-29",
        "sector": "XLK",
        "pct_above_ema20": 45.0,
        "pct_above_ema50": 35.0,
        "pct_above_ema200": 75.0,
        "pct_rs_positive": 45.0,
        "pct_momentum_positive": 35.0,
        "new_highs": 0,
        "new_lows": 0,
        "advances": 8,
        "declines": 16,
        "ad_net": -8,
        "n_valid_ema200": 20,
        "n_total": 20,
    }
    if with_n_valid_nhnl:
        row["n_valid_nhnl"] = 14
    return pd.DataFrame([row])


def test_breadth_header_con_n_valid_nhnl():
    out = "".join(render_sector_breadth(_breadth_df()))
    assert "N valid NH/NL" in out


def test_breadth_n_valid_nhnl_valor():
    out = "".join(render_sector_breadth(_breadth_df()))
    # 14 debe aparecer antes del Cob EMA200 100%
    assert "| 14 | 100% |" in out


def test_breadth_sin_columna_usa_nd():
    out = "".join(render_sector_breadth(_breadth_df(with_n_valid_nhnl=False)))
    assert "| N/D | 100% |" in out


def test_breadth_nota_n_valid_nhnl_aclaratoria():
    out = "".join(render_sector_breadth(_breadth_df()))
    assert "N valid NH/NL" in out
    assert ">=253 sesiones" in out
