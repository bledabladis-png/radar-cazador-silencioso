# tests/test_c3_ad_rendering.py
"""Tests C3 (2026-09-12): politica de presentacion de A/D Net.

B3: distinguir entre cero legitimo (mercado plano) y ausencia de
informacion direccional (N/D).

Contrato de _fmt_ad_net:
    advances/declines invalidos (NaN/None) -> N/D
    advances + declines == 0 -> N/D
    advances + declines > 0 -> formatear ad_net
"""
import numpy as np
import pandas as pd

from src.report.helpers import _fmt_ad_net
from src.report.sectorial import render_sector_breadth


# ============================================================
# T-C3-1: caso normal -> ad_net formateado
# ============================================================
def test_t_c3_1_caso_normal():
    assert _fmt_ad_net(13, 7, 6) == "+6"
    assert _fmt_ad_net(5, 15, -10) == "-10"


# ============================================================
# T-C3-2: mercado plano legitimo (5/5) -> +0 (NO N/D)
# ============================================================
def test_t_c3_2_mercado_plano_legitimo():
    assert _fmt_ad_net(5, 5, 0) == "+0"
    assert _fmt_ad_net(10, 10, 0) == "+0"


# ============================================================
# T-C3-3: sin informacion direccional (0/0) -> N/D
# ============================================================
def test_t_c3_3_sin_informacion_direccional():
    assert _fmt_ad_net(0, 0, 0) == "N/D"


# ============================================================
# T-C3-4: inputs invalidos (NaN/None) -> N/D
# ============================================================
def test_t_c3_4_inputs_invalidos():
    assert _fmt_ad_net(float('nan'), 7, None) == "N/D"
    assert _fmt_ad_net(13, float('nan'), None) == "N/D"
    assert _fmt_ad_net(None, None, None) == "N/D"
    assert _fmt_ad_net(pd.NA, 7, None) == "N/D"


# ============================================================
# T-C3-5: integracion en render_sector_breadth
# ============================================================
def test_t_c3_5_integracion_render():
    # Fila corrupta (tipo 2026-09-12): advances=0, declines=0, ad_net=0.
    df = pd.DataFrame([{
        'date': pd.Timestamp('2026-09-12'),
        'sector': 'XLB',
        'n_total': 27,
        'n_valid_ema200': 20,
        'pct_above_ema20': 15.0,
        'pct_above_ema50': 30.0,
        'pct_above_ema200': 60.0,
        'pct_rs_positive': 35.0,
        'pct_momentum_positive': 20.0,
        'count_accumulation': 4,
        'count_markup': 8,
        'count_distribution': 4,
        'count_markdown': 0,
        'new_highs': 1,
        'new_lows': 3,
        'advances': 0,
        'declines': 0,
        'unchanged': 20,
        'ad_net': 0,
        'advance_pct': np.nan,
    }])

    lines = render_sector_breadth(df)
    body = "\n".join(lines)

    # A/D debe ser N/D, no +0
    assert "| N/D |" in body, f"A/D debe ser N/D en fila corrupta.\n{body}"
    # El resto de campos debe seguir ahi (no alterar)
    assert "| XLB |" in body
    assert "| 15.0% |" in body  # pct_above_ema20
    assert "| 74% |" in body or "74%" in body  # cobertura
    # No debe aparecer "+0" para A/D
    assert "+0" not in body or body.count("+0") < 1, "no debe renderizar +0 en A/D"


# ============================================================
# T-C3-5b: fila con balance legitimo (5/5) -> +0 en render
# ============================================================
def test_t_c3_5b_render_balance_legitimo():
    df = pd.DataFrame([{
        'date': pd.Timestamp('2026-09-11'),
        'sector': 'XLK',
        'n_total': 75,
        'n_valid_ema200': 21,
        'pct_above_ema20': 50.0,
        'pct_above_ema50': 60.0,
        'pct_above_ema200': 85.0,
        'pct_rs_positive': 50.0,
        'pct_momentum_positive': 40.0,
        'count_accumulation': 2,
        'count_markup': 4,
        'count_distribution': 1,
        'count_markdown': 0,
        'new_highs': 0,
        'new_lows': 0,
        'advances': 5,
        'declines': 5,
        'unchanged': 11,
        'ad_net': 0,
        'advance_pct': 50.0,
    }])

    lines = render_sector_breadth(df)
    body = "\n".join(lines)

    assert "| +0 |" in body, f"debe renderizar +0 en balance real.\n{body}"