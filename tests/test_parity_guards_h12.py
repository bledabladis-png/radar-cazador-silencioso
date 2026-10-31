# -*- coding: utf-8 -*-
# Centinela H12: paridad de guards con alerts.py (H9) en 4 modulos.
# Cubre: synthesis, breadth, market_context, sectorial.

import pandas as pd

from src.report.synthesis import render_sintesis_senales
from src.report.breadth import render_breadth_market
from src.report.market_context import render_liderazgo_interno
from src.report.sectorial import render_sector_dispersion


def _real_nan(out):
    """Cuenta lineas con NAN en mayusculas o "nan%" - excluye "financiera"
    y "Financials" que contienen nan como substring."""
    hits = []
    for l in out.split(chr(10)):
        low = l.lower()
        # nan solo si aparece como token aislado o nan%
        if "nan%" in low or " nan " in low or low.endswith(" nan") or ": nan" in low:
            hits.append(l)
    return hits


def test_synthesis_breadth_none_no_crashea():
    out = "".join(render_sintesis_senales(
        "MIXED", "NEUTRAL", None,
        {"% sobre EMA200": None, "% sobre EMA20": None},
        {},
    ))
    assert not _real_nan(out)


def test_synthesis_div_none_no_crashea():
    out = "".join(render_sintesis_senales(
        "MIXED", "NEUTRAL", None, None,
        {"XLK": None, "XLF": {"status": "PRICE_STRONG_FLOW_UNCONFIRMED"}},
    ))
    assert not _real_nan(out)


def test_breadth_nan_pcts_sin_counts_no_crashea():
    """H12-bis: default de .get se evalua siempre, int(round(NaN*11)) petaba."""
    out = "".join(render_breadth_market({
        "% sobre EMA20": float("nan"),
        "% sobre EMA50": float("nan"),
        "% sobre EMA200": float("nan"),
        "New Highs (%)": float("nan"),
        "New Lows (%)": float("nan"),
    }))
    assert "0/11 (N/D)" in out
    assert not _real_nan(out)


def test_breadth_nan_con_counts_usa_counts():
    out = "".join(render_breadth_market({
        "% sobre EMA20": float("nan"), "% sobre EMA50": float("nan"),
        "% sobre EMA200": float("nan"), "New Highs (%)": float("nan"),
        "New Lows (%)": float("nan"),
        "EMA20 count": 5, "EMA50 count": 5, "EMA200 count": 5,
        "New Highs count": 2, "New Lows count": 1,
    }))
    assert "5/11 (N/D)" in out
    assert "2/11 (N/D)" in out


def test_market_context_nan_rs_no_crashea():
    df = pd.DataFrame([{
        "date": "2026-10-30", "sector": "XLK", "ticker": "AAPL",
        "price_ret_20d": 0.05, "rs_abs_20d": float("nan"),
        "rs_internal_20d": float("nan"), "classification": "X",
    }])
    out = "".join(render_liderazgo_interno(df))
    assert "N/D" in out
    assert not _real_nan(out)


def test_sectorial_dispersion_nan_no_crashea():
    df = pd.DataFrame([{
        "date": "2026-10-30", "sector": "XLK",
        "rs_p25": float("nan"), "rs_median": float("nan"), "rs_p75": float("nan"),
        "momentum_p25": float("nan"), "momentum_median": float("nan"),
        "momentum_p75": float("nan"),
    }])
    out = "".join(render_sector_dispersion(df))
    assert "N/D" in out
    assert not _real_nan(out)
