"""Test H13: el render de Liderazgo relativo interno declara el
benchmark exacto (SPY), la fuente del precio y la formula.

Auditoria externa 2026-10-15, hallazgo H13 (medio): el auditor cruzo
PLTR con ^GSPC y veia 15.16% en lugar de 15.27%. No es bug: el
benchmark de rs_abs_20d es SPY (no ^GSPC). El render no lo declaraba.
"""
import pandas as pd

from src.report.market_context import render_liderazgo_interno


def _df():
    return pd.DataFrame([{
        "date": "2026-10-13",
        "sector": "XLK",
        "ticker": "AAPL",
        "price_ret_20d": 0.10,
        "sector_ret_20d": 0.05,
        "benchmark_ret_20d": 0.03,
        "rs_abs_20d": (1.10/1.03) - 1,
        "rs_internal_20d": (1.10/1.05) - 1,
        "classification": "Liderazgo relativo doble",
    }])


def test_h13_declara_benchmark_spy():
    out = "".join(render_liderazgo_interno(_df()))
    assert "SPY" in out
    assert "benchmark" in out.lower()


def test_h13_declara_formula():
    out = "".join(render_liderazgo_interno(_df()))
    assert "(1+ret20_ticker)" in out or "(1+ret20" in out
    assert "/" in out


def test_h13_declara_fuente():
    out = "".join(render_liderazgo_interno(_df()))
    assert "Yahoo" in out
    assert "Close" in out or "close" in out


def test_h13_aclara_gspc_vs_spy():
    out = "".join(render_liderazgo_interno(_df()))
    # Debe aclarar que discrepancias con ^GSPC son de fuente, no de calculo
    assert "GSPC" in out
    assert "fuente" in out.lower()
