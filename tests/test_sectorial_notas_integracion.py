"""Integración: nota A/D de breadth refleja n_universo_confirmation real.

Bug 2026-11-08: report_generator.py:145 leia confirmation_data["ad_net"]
pero el productor (mte_confirmation.py:122) emite confirmation_data["ad"].
n_universo_confirmation llegaba siempre None y la nota caia al elif sin
numero. Este test verifica el cableado end-to-end.
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


def test_estructura_confirmation_data_ad_es_clave_real():
    """mte_confirmation emite confirmation_data['ad'], no 'ad_net'."""
    src = Path("src/pipeline/mte_confirmation.py").read_text(encoding="utf-8")
    assert "confirmation_data['ad'] = ad_data" in src, (
        "productor cambio de clave; actualizar report_generator.py"
    )
    assert "confirmation_data['ad_net']" not in src, (
        "productor ya no usa 'ad_net'; limpiar consumidores"
    )


def test_consumidor_lee_clave_ad():
    """report_generator.py debe leer 'ad', no 'ad_net'."""
    src = Path("src/report_generator.py").read_text(encoding="utf-8")
    assert 'confirmation_data or {}).get("ad", {})' in src or \
           "confirmation_data or {}).get('ad', {})" in src, (
        "report_generator no lee 'ad': bug 2026-11-08"
    )
    assert 'confirmation_data or {}).get("ad_net"' not in src, (
        "report_generator sigue leyendo 'ad_net' (clave inexistente)"
    )


def test_render_nota_con_n_universo_real():
    """Con n_universo_confirmation=713, la nota debe decir '713 tickers'."""
    df = _snap()
    out = "".join(render_sector_breadth(df, n_universo_confirmation=713))
    assert "universo completo, 713 tickers" in out
    assert "universo completo del pipeline" not in out