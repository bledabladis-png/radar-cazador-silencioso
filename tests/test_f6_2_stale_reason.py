"""F6-2 (2026-09-28): stale_reason distingue MARKET_CLOSED vs DATA_PENDING.

F6-01: el render imprimia "mercado cerrado" para cualquier stale. Ahora
distingue si la causa es fin de semana/festivo o sesion pendiente de
datos en stock_prices.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.report.sectorial import render_sector_breadth


def _make_snap(date="2026-09-24"):
    return pd.DataFrame({
        "date": [date],
        "sector": ["XLK"],
        "n_total": [20],
        "n_valid_ema20": [20],
        "n_valid_ema50": [20],
        "n_valid_ema200": [20],
        "n_valid_nhnl": [20],
        "n_valid_momentum": [20],
        "n_valid_ad": [20],
        "pct_above_ema20": [80.0],
        "pct_above_ema50": [75.0],
        "pct_above_ema200": [70.0],
        "pct_rs_positive": [60.0],
        "pct_momentum_positive": [55.0],
        "count_accumulation": [10],
        "count_markup": [5],
        "count_distribution": [3],
        "count_markdown": [2],
        "new_highs": [1],
        "new_lows": [0],
        "advances": [12],
        "declines": [8],
        "ad_net": [4],
    })


def test_stale_reason_market_closed_muestra_mercado_cerrado():
    lines = render_sector_breadth(_make_snap(), is_stale=True,
                                   stale_reason="MARKET_CLOSED")
    joined = "".join(lines)
    assert "mercado cerrado" in joined
    assert "WARN" not in joined


def test_stale_reason_data_pending_muestra_warn_y_verificar():
    lines = render_sector_breadth(_make_snap(), is_stale=True,
                                   stale_reason="DATA_PENDING")
    joined = "".join(lines)
    assert "[WARN]" in joined
    assert "stock_prices" in joined
    assert "No es festivo" in joined


def test_stale_reason_error_muestra_warn():
    lines = render_sector_breadth(_make_snap(), is_stale=True,
                                   stale_reason="ERROR")
    joined = "".join(lines)
    assert "[WARN]" in joined
    assert "error en el calculo" in joined


def test_stale_reason_none_mantiene_texto_legacy():
    """Compatibilidad: stale_reason=None usa el texto historico."""
    lines = render_sector_breadth(_make_snap(), is_stale=True, stale_reason=None)
    joined = "".join(lines)
    assert "mercado cerrado" in joined


def test_is_stale_false_no_muestra_aviso():
    lines = render_sector_breadth(_make_snap(), is_stale=False)
    joined = "".join(lines)
    assert "Sin actualizacion" not in joined