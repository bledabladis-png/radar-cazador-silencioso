# -*- coding: utf-8 -*-
"""D29 (2026-09-30): tests para zonas no cubiertas de 3 render modules.

- src/report/sector_context.py: 63% -> cubre las ramas de iteracion
  (render_matriz_regimen, render_wyckoff_sectorial, render_momentum_amplitud).
- src/report/sectorial.py: 67% -> render_sector_concentration y
  render_sector_dispersion (ramas de iteracion + nota final).
- src/report/synthesis.py: 66% -> render_indices_internacionales con
  leaders, render_sintesis_senales con MIXED + dispersion + lider SLPM
  + divergencias.

Los tests existentes (test_render_representatividad_lider,
test_render_aclaraciones_d2c_d3_e1_a1, etc.) cubren el camino vacio.
Aqui van los caminos con datos.
"""
import pandas as pd

from src.report.sector_context import (
    render_matriz_regimen,
    render_wyckoff_sectorial,
    render_momentum_amplitud,
)
from src.report.sectorial import (
    render_sector_concentration,
    render_sector_dispersion,
)
from src.report.synthesis import (
    render_indices_internacionales,
    render_sintesis_senales,
)


# ============================================================
# sector_context.render_matriz_regimen
# ============================================================
def test_matriz_regimen_con_datos():
    df = pd.DataFrame([{
        "sector": "XLK",
        "price_ret_20d": 0.05,
        "pct_above_ema50": 60.0,
        "flow_20d_sum": 1_500_000.0,
        "wyckoff_phase": "MARKUP",
        "positive_conditions": 4,
        "regime_reading": "Favorable",
    }])
    out = render_matriz_regimen(df)
    joined = "".join(out)
    assert "## Matriz" in joined
    assert "XLK" in joined
    assert "MARKUP" in joined


def test_matriz_regimen_vacio():
    assert render_matriz_regimen(None) == []
    assert render_matriz_regimen(pd.DataFrame()) == []


# ============================================================
# sector_context.render_wyckoff_sectorial
# ============================================================
def test_wyckoff_sectorial_con_datos():
    df = pd.DataFrame([{
        "date": "2026-09-30",
        "sector": "XLK",
        "pct_accumulation": 10.0,
        "pct_markup": 40.0,
        "pct_range": 30.0,
        "pct_distribution": 15.0,
        "pct_markdown": 5.0,
        "n_valid_wyckoff": 20,
        "coverage_wyckoff": 100.0,
    }])
    out = render_wyckoff_sectorial(df)
    joined = "".join(out)
    assert "## Distribuci" in joined
    assert "XLK" in joined


def test_wyckoff_sectorial_multiples_fechas_usa_ultima():
    """Con 2 fechas, solo la mas reciente aparece.

    El sector es XLK en ambas filas. La fila del 2026-09-28 tiene
    pct_range=100, pct_markup=0. La del 2026-09-30 tiene markup=50,
    range=0. Si el filtro funciona, la fila del 28 no debe estar.
    """
    df = pd.DataFrame([
        {"date": "2026-09-28", "sector": "XLK",
         "pct_accumulation": 0, "pct_markup": 0, "pct_range": 100,
         "pct_distribution": 0, "pct_markdown": 0,
         "n_valid_wyckoff": 20, "coverage_wyckoff": 100.0},
        {"date": "2026-09-30", "sector": "XLK",
         "pct_accumulation": 50, "pct_markup": 50, "pct_range": 0,
         "pct_distribution": 0, "pct_markdown": 0,
         "n_valid_wyckoff": 20, "coverage_wyckoff": 100.0},
    ])
    out = render_wyckoff_sectorial(df)
    joined = "".join(out)
    # La fila de la ultima fecha tiene markup=50.
    assert "| 50% | 50% |" in joined
    # La fila de la fecha antigua tiene range=100 y markup=0.
    # Firma unica: "| 0% | 0% | 100% | 0% | 0% |"
    firma_vieja = "| 0% | 0% | 100% | 0% | 0% |"
    assert firma_vieja not in joined, (
        "aparece la fila de la fecha antigua:\n" + joined)


# ============================================================
# sector_context.render_momentum_amplitud
# ============================================================
def test_momentum_amplitud_con_datos():
    df = pd.DataFrame([{
        "date": "2026-09-30",
        "sector": "XLE",
        "delta_1d_ema20": 1.5,
        "delta_5d_ema20": 3.0,
        "delta_20d_ema20": 5.0,
        "delta_5d_ema50": 2.0,
        "delta_5d_ema200": 4.0,
        "breadth_expansion_5d": 60.0,
        "breadth_deterioration_5d": 40.0,
    }])
    out = render_momentum_amplitud(df)
    joined = "".join(out)
    assert "## Momentum de amplitud" in joined
    assert "XLE" in joined


# ============================================================
# sectorial.render_sector_concentration
# ============================================================
def _conc_row():
    return {
        "date": "2026-09-30",
        "sector": "XLK",
        "top1_positive_return_concentration": 0.5,
        "top3_positive_return_concentration": 0.7,
        "top5_positive_return_concentration": 0.9,
        "rs_median": 0.1,
        "momentum_median": 0.02,
        "flow_median": 1.5,
        "wyckoff_median": 0.6,
        "wls_median": 0.8,
        "leader_ticker": "NVDA",
        "leader_return20": 0.15,
        "coverage_rs": 100.0,
        "coverage_momentum": 100.0,
        "coverage_flow": 100.0,
        "coverage_wyckoff": 95.0,
        "coverage_wls": 95.0,
    }


def test_sector_concentration_con_datos():
    df = pd.DataFrame([_conc_row()])
    out = render_sector_concentration(df)
    joined = "".join(out)
    assert "## Concentraci" in joined
    assert "XLK" in joined
    assert "NVDA" in joined


def test_sector_concentration_nota_final():
    df = pd.DataFrame([_conc_row()])
    joined = "".join(render_sector_concentration(df))
    assert "Criterio 'Lider'" in joined


# ============================================================
# sectorial.render_sector_dispersion
# ============================================================
def _disp_row():
    return {
        "date": "2026-09-30",
        "sector": "XLF",
        "rs_p25": -0.05,
        "rs_median": 0.10,
        "rs_p75": 0.25,
        "momentum_p25": -0.02,
        "momentum_median": 0.01,
        "momentum_p75": 0.04,
    }


def test_sector_dispersion_con_datos():
    df = pd.DataFrame([_disp_row()])
    out = render_sector_dispersion(df)
    joined = "".join(out)
    assert "## Dispersi" in joined
    assert "XLF" in joined


def test_sector_dispersion_nota_percentiles():
    df = pd.DataFrame([_disp_row()])
    joined = "".join(render_sector_dispersion(df))
    assert "percentiles de Flow" in joined


# ============================================================
# synthesis.render_indices_internacionales
# ============================================================
def test_indices_internacionales_con_phases_y_leaders():
    phases = {"DAX 40": "MARKUP", "FTSE 100": "ACCUMULATION"}
    leaders = {
        "DAX 40": pd.DataFrame([{
            "ticker": "SIE.DE", "rs": 1.05, "rs_mom": 0.03,
            "flow_proxy_z": 1.5, "wls": 1.2, "wyckoff_phase": "MARKUP",
        }]),
    }
    out = render_indices_internacionales(phases, leaders)
    joined = "".join(out)
    assert "DAX 40" in joined
    assert "SIE.DE" in joined


def test_indices_internacionales_sin_leaders():
    phases = {"DAX 40": "MARKUP"}
    out = render_indices_internacionales(phases, None)
    joined = "".join(out)
    assert "Ning" in joined or "ning" in joined.lower()


def test_indices_internacionales_leaders_vacio():
    """Leaders con DataFrame vacio -> skip, no crashea."""
    leaders = {"DAX 40": pd.DataFrame()}
    out = render_indices_internacionales({}, leaders)
    assert isinstance(out, list)


# ============================================================
# synthesis.render_sintesis_senales
# ============================================================
def test_sintesis_senales_recesion():
    out = render_sintesis_senales(
        "RECESSION", None, "NEUTRAL", None, None, None)
    joined = "".join(out)
    assert "RECESSION" in joined
    assert "estres elevado" in joined or "estr" in joined.lower()


def test_sintesis_senales_mixed_con_dispersion():
    """Rama MIXED: lee dispersion_reading del ultimo dia."""
    disp_df = pd.DataFrame([{"dispersion_reading": "MODERADA"}])
    out = render_sintesis_senales(
        "MIXED", None, "NEUTRAL", disp_df, None, None)
    joined = "".join(out)
    assert "MIXED" in joined
    assert "moderada" in joined.lower()


def test_sintesis_senales_mixed_sin_dispersion():
    out = render_sintesis_senales(
        "MIXED", None, "NEUTRAL", None, None, None)
    joined = "".join(out)
    assert "variable" in joined.lower()


def test_sintesis_senales_lider_slpm_confirmado():
    slpm = {"sector": "XLK", "state": "CONFIRMED"}
    out = render_sintesis_senales(
        "EXPANSION", slpm, "NEUTRAL", None, None, None)
    joined = "".join(out)
    assert "XLK" in joined
    assert "CONFIRMED" in joined


def test_sintesis_senales_lider_slpm_unresolved():
    slpm = {"sector": "XLK", "state": "UNRESOLVED"}
    out = render_sintesis_senales(
        "MIXED", slpm, "NEUTRAL", None, None, None)
    joined = "".join(out)
    assert "UNRESOLVED" in joined
    assert "no confirmado" in joined.lower() or "no confirm" in joined.lower()


def test_sintesis_senales_liquidez_alta():
    out = render_sintesis_senales(
        "RECESSION", None, "HIGH_STRESS", None, None, None)
    joined = "".join(out)
    assert "HIGH_STRESS" in joined


def test_sintesis_senales_divergencias_price_flow():
    divergences = {
        "XLK": {"status": "PRICE_STRONG_FLOW_UNCONFIRMED"},
    }
    out = render_sintesis_senales(
        "MIXED", None, "NEUTRAL", None, None, divergences)
    assert isinstance(out, list)
