# -*- coding: utf-8 -*-
"""2026-10-30: -0.00 en Contexto transversal de mercado.

El renderer local usaba f"{v:.2f}" directo. mean_corr ~ -1e-17 se
formateaba como "-0.00" (XLE 60d FX en el run 41037457375). Ahora usa
_fmt_signed (FU-003a): cero visual sin signo, no-cero con signo.
"""
import pandas as pd

from src.report.market_context import render_contexto_cross_asset


def _df(rows):
    """rows: list of (sector, window, asset_class, mean_corr)."""
    return pd.DataFrame(
        [{"sector": s, "window": w, "asset_class": ac, "mean_corr": mc}
         for s, w, ac, mc in rows]
    )


def test_neg_zero_se_renderiza_como_cero_sin_signo():
    """-1e-17 en FX -> "0.00", no "-0.00"."""
    df = _df([
        ("XLE", 60, "fx", -1e-17),
        ("XLE", 60, "equity", 0.30),
        ("XLE", 60, "rates", 0.43),
        ("XLE", 60, "credit", 0.43),
        ("XLE", 60, "commodities", 0.11),
        ("XLE", 60, "volatility", -0.28),
    ])
    out = "".join(render_contexto_cross_asset(df))
    assert "-0.00" not in out
    assert "| 0.00 |" in out


def test_valor_no_cero_con_signo():
    """Valores no cero mantienen el signo."""
    df = _df([
        ("XLE", 20, "equity", -0.38),
        ("XLE", 20, "rates", -0.15),
        ("XLE", 20, "credit", -0.48),
        ("XLE", 20, "commodities", 0.20),
        ("XLE", 20, "fx", -0.08),
        ("XLE", 20, "volatility", 0.40),
    ])
    out = "".join(render_contexto_cross_asset(df))
    assert "-0.38" in out
    assert "+0.20" in out
    assert "+0.40" in out


def test_nan_sigue_siendo_nd():
    df = _df([
        ("XLE", 60, "fx", float("nan")),
        ("XLE", 60, "equity", 0.30),
        ("XLE", 60, "rates", 0.10),
        ("XLE", 60, "credit", 0.20),
        ("XLE", 60, "commodities", 0.05),
        ("XLE", 60, "volatility", 0.15),
    ])
    out = "".join(render_contexto_cross_asset(df))
    assert "N/D" in out
