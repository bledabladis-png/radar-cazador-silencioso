# -*- coding: utf-8 -*-
# Centinela C1-C6: confirmation.py con NaN en todos los campos numericos.

from src.report.confirmation import render_confirmation


def _full_nan():
    return {
        "t10y3m": float("nan"), "rv_21d": float("nan"), "rv_60d": float("nan"),
        "vrp_21d": float("nan"), "vrp_60d": float("nan"),
        "fls": {"fls_normalized": float("nan"), "components": 0,
                "total_components": 5, "stressed_components": 0, "detail": {}},
        "ad": {"ad_net": float("nan"), "advances": float("nan"),
               "declines": float("nan"), "new_highs": float("nan"),
               "new_lows": float("nan"), "nh_nl": float("nan"),
               "breadth_thrust": float("nan"), "ad_line": float("nan")},
        "ratios": {"copper_gold": float("nan")},
    }


def test_todo_nan_sin_nan_literal():
    out = "".join(render_confirmation(_full_nan()))
    lineas_con_nan = [l for l in out.split("\n") if "nan" in l.lower()]
    assert not lineas_con_nan, "NaN literal: " + str(lineas_con_nan)


def test_t10y3m_nan_no_pinta_spread():
    out = "".join(render_confirmation(_full_nan()))
    assert "10Y-3M Spread" not in out


def test_fls_nan_pinta_nd():
    out = "".join(render_confirmation(_full_nan()))
    assert "Funding & Liquidity Stress (FLS):** N/D/100" in out


def test_ad_conteos_nan_pinta_nd():
    out = "".join(render_confirmation(_full_nan()))
    assert "N/D avances / N/D descensos" in out
    assert "N/D maximos / N/D minimos" in out


def test_ad_line_nan_pinta_nd():
    out = "".join(render_confirmation(_full_nan()))
    assert "A/D Line (acumulada):** N/D" in out


def test_ratio_val_nan_pinta_nd():
    out = "".join(render_confirmation(_full_nan()))
    assert "| Copper/Gold | N/D |" in out


def test_valores_normales_intactos():
    data = {
        "t10y3m": -0.42,
        "fls": {"fls_normalized": 0.85, "components": 5, "total_components": 5,
                "stressed_components": 4, "detail": {}},
        "ad": {"ad_net": 120, "advances": 300, "declines": 180,
               "new_highs": 45, "new_lows": 12, "nh_nl": 33,
               "breadth_thrust": 0.55, "ad_line": 12345.0},
        "ratios": {"copper_gold": 0.001234},
    }
    out = "".join(render_confirmation(data))
    assert "-0.42%" in out
    assert "85/100" in out
    assert "300 avances / 180 descensos" in out
    assert "45 maximos / 12 minimos" in out
    assert "12345" in out
    assert "0.0012" in out
