# -*- coding: utf-8 -*-
# Centinela B6-H9: alerts.py con None/valores desconocidos.

from src.report.alerts import render_alerts, render_cross_module


def test_ema_none_sin_crash():
    out = "".join(render_alerts(
        {"% sobre EMA200": None, "% sobre EMA20": None}, "NEUTRAL", {}))
    assert "Breadth Divergence" not in out


def test_ema_nan_sin_crash():
    out = "".join(render_alerts(
        {"% sobre EMA200": float("nan"), "% sobre EMA20": float("nan")}, "NEUTRAL", {}))
    assert "Breadth Divergence" not in out


def test_ema_valores_divergentes_pinta_alerta():
    out = "".join(render_alerts(
        {"% sobre EMA200": 0.75, "% sobre EMA20": 0.50}, "NEUTRAL", {}))
    assert "Breadth Divergence" in out


def test_div_none_sin_crash():
    out = "".join(render_alerts({}, "NEUTRAL", {"XLK": None}))
    assert "XLK" not in out


def test_div_valido_pinta_alerta():
    out = "".join(render_alerts(
        {}, "NEUTRAL", {"XLK": {"status": "PRICE_WEAK", "message": "test msg"}}))
    assert "test msg" in out


def test_conflict_level_desconocido_warn():
    out = "".join(render_cross_module(
        {"conflict_level": "NEW_LEVEL", "message": "x"}))
    assert "[WARN] cross_module" in out
    assert "NEW_LEVEL" in out


def test_conflict_level_conocido_sin_warn():
    for level in ["CONSENSUS", "CONFLICT", "DIVERGENCE", "MIXED"]:
        out = "".join(render_cross_module(
            {"conflict_level": level, "message": "x"}))
        assert "[WARN] cross_module" not in out, level
