# -*- coding: utf-8 -*-
"""FIX-BREADTH-WALKBACK (2026-11-14): mensaje de stale segun contexto.

Contrato:
- stale_reason != 'DATA_PENDING': resto de ramas intactas.
- stale_reason == 'DATA_PENDING' + walk_back_applied=False:
  WARN "sesion NYSE pendiente ... verificar cobertura" (anomalia real).
- stale_reason == 'DATA_PENDING' + walk_back_applied=True + sin
  worst_market: sin WARN, texto informativo "walk-back aplicado".
- stale_reason == 'DATA_PENDING' + worst_market presente: WARN
  cobertura especifica (walk_back NO la silencia).
"""
import pandas as pd

from src.report.sectorial import render_sector_breadth


def _df(sector="XLK"):
    return pd.DataFrame([{
        "date": pd.Timestamp("2026-11-12"),
        "sector": sector,
        "n_total": 20,
        "n_valid_ema200": 20,
        "pct_above_ema20": 50.0,
        "pct_above_ema50": 50.0,
        "pct_above_ema200": 50.0,
        "pct_rs_positive": 50.0,
        "pct_momentum_positive": 50.0,
        "new_highs": 0,
        "new_lows": 0,
        "advances": 10,
        "declines": 10,
        "ad_net": 0,
        "n_valid_nhnl": 20,
    }])


def test_sin_walk_back_mantiene_warn_generico():
    out = "".join(render_sector_breadth(
        _df(), is_stale=True, stale_reason="DATA_PENDING",
        stale_context=None, walk_back_applied=False))
    assert "[WARN]" in out
    assert "sesion NYSE pendiente" in out
    assert "verificar cobertura" in out
    assert "walk-back aplicado" not in out


def test_walk_back_sin_peor_mercado_sin_alarma():
    out = "".join(render_sector_breadth(
        _df(), is_stale=True, stale_reason="DATA_PENDING",
        stale_context=None, walk_back_applied=True))
    assert "[WARN]" not in out
    assert "verificar cobertura" not in out
    assert "sesion NYSE pendiente" not in out
    assert "walk-back aplicado" in out
    assert "2026-11-12" in out


def test_walk_back_con_peor_mercado_mantiene_cobertura_especifica():
    out = "".join(render_sector_breadth(
        _df(), is_stale=True, stale_reason="DATA_PENDING",
        stale_context={"worst_market": "LSE", "worst_coverage": 0.85},
        walk_back_applied=True))
    assert "[WARN]" in out
    assert "LSE" in out
    assert "85.0%" in out
    assert "walk-back aplicado" not in out


def test_market_closed_mantiene_texto_historico():
    out = "".join(render_sector_breadth(
        _df(), is_stale=True, stale_reason="MARKET_CLOSED",
        stale_context=None, walk_back_applied=False))
    assert "mercado cerrado" in out
    assert "[WARN]" not in out


def test_no_stale_no_muestra_warn():
    out = "".join(render_sector_breadth(
        _df(), is_stale=False, stale_reason=None,
        stale_context=None, walk_back_applied=False))
    assert "Sin actualizacion" not in out
    assert "Datos al 2026-11-12" in out
