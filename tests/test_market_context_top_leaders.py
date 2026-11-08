"""Centinela: TOP_LEADERS_PER_SECTOR unica fuente del '5' de liderazgo interno.

2026-11-07: el numero de lideres por sector estaba hardcodeado en
.nlargest(5) y en el texto "los 5 tickers". Extraido a constante.
Si cambia la constante, el texto cambia sin tocar el render.
"""
import importlib
from pathlib import Path

import pandas as pd

import src.report.market_context as mc


def _make_rs_df(n_sectores=2, n_tickers_por_sector=10):
    rows = []
    for s in range(n_sectores):
        for t in range(n_tickers_por_sector):
            rows.append({
                "date": "2026-11-06",
                "sector": f"S{s:02d}",
                "ticker": f"T{s:02d}{t:02d}",
                "price_ret_20d": 1.0 - 0.01 * t,
                "rs_abs_20d": 0.0,
                "rs_internal_20d": 0.0,
                "classification": "NEUTRAL",
            })
    return pd.DataFrame(rows)


def test_constante_existe():
    assert hasattr(mc, "TOP_LEADERS_PER_SECTOR")
    assert isinstance(mc.TOP_LEADERS_PER_SECTOR, int)
    assert mc.TOP_LEADERS_PER_SECTOR >= 1


def test_nlargest_usa_constante_no_literal():
    src = Path("src/report/market_context.py").read_text(encoding="utf-8")
    assert "nlargest(5," not in src, (
        "nlargest(5,...) hardcodeado: debe usar TOP_LEADERS_PER_SECTOR"
    )
    assert "nlargest(TOP_LEADERS_PER_SECTOR," in src


def test_nota_deriva_numero_de_constante():
    df = _make_rs_df(n_sectores=2, n_tickers_por_sector=10)
    importlib.reload(mc)
    out = "".join(mc.render_liderazgo_interno(df))
    n = mc.TOP_LEADERS_PER_SECTOR
    assert f"los {n} tickers" in out, (
        f"la nota debe decir 'los {n} tickers' derivado de la constante"
    )


def test_nota_cambia_si_cambia_constante(monkeypatch):
    importlib.reload(mc)
    monkeypatch.setattr(mc, "TOP_LEADERS_PER_SECTOR", 3)
    df = _make_rs_df(n_sectores=2, n_tickers_por_sector=10)
    out = "".join(mc.render_liderazgo_interno(df))
    assert "los 3 tickers" in out, (
        "si la constante cambia a 3, la nota debe reflejarlo"
    )