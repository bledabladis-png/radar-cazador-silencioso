"""Centinela: nota A/D de breadth deriva sus números del df.

2026-11-07: el literal "~220 tickers" y "~313 tickers" se sustituyo
por f-string con n_top20 (sum de n_total) y n_universo_confirmation
(parametro derivado de confirmation_data). Este test verifica que los
numeros no reaparezcan hardcodeados y que la firma sigue derivando.
"""
from pathlib import Path

import pandas as pd

from src.report.sectorial import render_sector_breadth


def _snap(n_sectores=11, n_total=20):
    rows = []
    for i in range(n_sectores):
        rows.append({
            "date": "2026-11-06",
            "sector": f"S{i:02d}",
            "n_total": n_total,
            "n_valid_ema200": n_total,
            "n_valid_nhnl": n_total,
            "pct_above_ema20": 0.5, "pct_above_ema50": 0.5,
            "pct_above_ema200": 0.5, "pct_rs_positive": 0.5,
            "pct_momentum_positive": 0.5,
            "new_highs": 0, "new_lows": 0,
            "advances": 0, "declines": 0, "ad_net": 0,
        })
    return pd.DataFrame(rows)


def test_nota_ad_deriva_n_top20_de_df():
    df = _snap(11, 20)
    out = "".join(render_sector_breadth(df))
    assert "suma total = 220" in out, (
        "11 sectores x 20 = 220, debe aparecer derivado"
    )


def test_nota_ad_deriva_n_top20_si_cambia_n_sectores():
    df = _snap(15, 20)
    out = "".join(render_sector_breadth(df))
    assert "suma total = 300" in out, (
        "15 sectores x 20 = 300, el numero debe cambiar"
    )


def test_nota_ad_incluye_n_universo_confirmation():
    df = _snap(11, 20)
    out = "".join(render_sector_breadth(df, n_universo_confirmation=713))
    assert "universo completo, 713 tickers" in out


def test_nota_ad_sin_n_confirmation_no_inventa_numero():
    df = _snap(11, 20)
    out = "".join(render_sector_breadth(df))
    assert "suma total = 220" in out
    assert "universo completo del pipeline" in out
    # No debe aparecer "~313" ni "713" sin parametro
    assert "~313" not in out
    assert "713" not in out


def test_sin_literales_220_313():
    src = Path("src/report/sectorial.py").read_text(encoding="utf-8")
    assert "~220 tickers" not in src
    assert "~313 tickers" not in src