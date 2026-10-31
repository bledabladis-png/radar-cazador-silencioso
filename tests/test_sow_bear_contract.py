# -*- coding: utf-8 -*-
"""Tests de contrato SOW-BEAR-CONFIRMATION (doc 57, v1.10.1).

T1-T10 del contrato. TDD: se escriben antes del wiring a produccion.
El modulo indicators/sow_bear.py es puro y testeable aislado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.sow_bear import (
    compute_sow_bear_confirmation,
    count_sow_in_window,
    is_bear,
    M_SOW_WINDOW,
)


def _make_market(prices, tickers=None, start="2024-01-01"):
    """DataFrame MultiIndex (field, ticker) con Close + Volume."""
    n = len(prices)
    idx = pd.date_range(start, periods=n, freq="B")
    if tickers is None:
        tickers = ["SPY"]
    data = {}
    for i, t in enumerate(tickers):
        p = np.asarray(prices, dtype=float) if i == 0 else np.asarray(prices, dtype=float) * 100
        data[("Close", t)] = p
        data[("Volume", t)] = np.full(n, 1e6)
    return pd.DataFrame(data, index=idx)


def _make_holdings(n_valid=6, n_with_sow=3, sow_offset_sessions=0, include_event_date=True):
    """Lista de dicts como los produce _evaluate_holding."""
    out = []
    for i in range(n_valid):
        has_sow = i < n_with_sow
        out.append({
            "ticker": f"T{i}",
            "sow_status": "CONFIRMED" if has_sow else "NOT_OBSERVED",
            "sow_event_date": f"OFFSET_{sow_offset_sessions}" if (has_sow and include_event_date) else None,
            "evaluable": True,
        })
    return out


def _market_with_bear_at(as_of_idx, total=300, base=400.0, crash_pct=0.20):
    """Genera serie de precios que en as_of_idx esta en bear."""
    prices = np.linspace(base, base, total)
    # caida sostenida desde 2/3 del camino
    crash_start = total // 3
    for i in range(crash_start, total):
        factor = 1.0 - crash_pct * (i - crash_start) / (total - crash_start)
        prices[i] = base * factor
    return prices


def _market_without_bear(total=300, base=400.0):
    """Serie alcista suave, sin bear."""
    return np.linspace(base, base * 1.2, total)


# --- T1: bear=False -> NOT_BEAR aunque haya SOW ---

def test_t1_bear_false_devuelve_not_bear():
    df = _make_market(_market_without_bear())
    as_of = df.index[-1]
    holdings = _make_holdings(n_valid=6, n_with_sow=5)
    # convertir OFFSET_0 a fecha real (ultima sesion)
    for h in holdings:
        if h["sow_event_date"] == "OFFSET_0":
            h["sow_event_date"] = as_of
    r = compute_sow_bear_confirmation(df, holdings, as_of)
    assert r["sow_bear_confirmation"] is False
    assert r["sow_bear_reason"] == "NOT_BEAR"
    assert r["sow_bear_hits"] == 0


# --- T2: bear=True, 0 SOW ---

def test_t2_bear_true_sin_sow_devuelve_insufficient_hits():
    df = _make_market(_market_with_bear_at(299))
    as_of = df.index[-1]
    holdings = _make_holdings(n_valid=6, n_with_sow=0)
    r = compute_sow_bear_confirmation(df, holdings, as_of)
    assert r["sow_bear_confirmation"] is False
    assert r["sow_bear_reason"] == "INSUFFICIENT_SOW_HITS"
    assert r["sow_bear_hits"] == 0


# --- T3: bear=True, 2 SOW < N=3 ---

def test_t3_bear_true_2_sow_devuelve_insufficient_hits():
    df = _make_market(_market_with_bear_at(299))
    as_of = df.index[-1]
    holdings = _make_holdings(n_valid=6, n_with_sow=2)
    for h in holdings:
        if h["sow_event_date"] == "OFFSET_0":
            h["sow_event_date"] = as_of
    r = compute_sow_bear_confirmation(df, holdings, as_of)
    assert r["sow_bear_confirmation"] is False
    assert r["sow_bear_reason"] == "INSUFFICIENT_SOW_HITS"
    assert r["sow_bear_hits"] == 2


# --- T4: bear=True, 3 SOW, 6 validos -> OK ---

def test_t4_bear_true_3_sow_ok():
    df = _make_market(_market_with_bear_at(299))
    as_of = df.index[-1]
    holdings = _make_holdings(n_valid=6, n_with_sow=3)
    for h in holdings:
        if h["sow_event_date"] == "OFFSET_0":
            h["sow_event_date"] = as_of
    r = compute_sow_bear_confirmation(df, holdings, as_of)
    assert r["sow_bear_confirmation"] is True
    assert r["sow_bear_reason"] == "OK"
    assert r["sow_bear_hits"] == 3


# --- T5: bear=True, 3 SOW, solo 3 validos -> INSUFFICIENT_COVERAGE ---

def test_t5_bear_true_3_sow_pocos_validos_insufficient_coverage():
    df = _make_market(_market_with_bear_at(299))
    as_of = df.index[-1]
    holdings = _make_holdings(n_valid=3, n_with_sow=3)
    for h in holdings:
        if h["sow_event_date"] == "OFFSET_0":
            h["sow_event_date"] = as_of
    r = compute_sow_bear_confirmation(df, holdings, as_of)
    assert r["sow_bear_confirmation"] is False
    assert r["sow_bear_reason"] == "INSUFFICIENT_COVERAGE"
    assert r["sow_bear_hits"] == 3


# --- T6: SPY ausente -> NO_DATA_MARKET_PROXY ---

def test_t6_spy_ausente_no_data_market_proxy():
    df = _make_market(_market_without_bear(), tickers=["AAPL"])
    as_of = df.index[-1]
    holdings = _make_holdings(n_valid=6, n_with_sow=3)
    r = compute_sow_bear_confirmation(df, holdings, as_of)
    assert r["sow_bear_confirmation"] is False
    assert r["sow_bear_reason"] == "NO_DATA_MARKET_PROXY"
    assert r["sow_bear_hits"] == 0


# --- T7: invariante I1 (clasificacion de fase no se modifica) ---
# Cubierto estructuralmente: el modulo no importa classify_wyckoff_phase.
# Test defensivo: si alguien lo importa, el AST del test de contrato falla.

def test_t7_modulo_no_importa_clasificacion_de_fase():
    import ast
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "indicators" / "sow_bear.py"
    tree = ast.parse(p.read_text(encoding="utf-8"))
    prohibidos = {"classify_wyckoff_phase", "classify_phase_at", "classify_phase"}
    encontrados = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in prohibidos:
                    encontrados.add(alias.name)
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in prohibidos:
                encontrados.add(f.id)
            if isinstance(f, ast.Attribute) and f.attr in prohibidos:
                encontrados.add(f.attr)
    assert not encontrados, (
        "sow_bear.py no debe llamar a funciones de clasificacion de "
        "fase (invariante I1). Encontrados: {0}".format(encontrados)
    )


# --- T8: invariante I4 (determinismo) ---

def test_t8_determinismo_dos_llamadas():
    df = _make_market(_market_with_bear_at(299))
    as_of = df.index[-1]
    holdings = _make_holdings(n_valid=6, n_with_sow=3)
    for h in holdings:
        if h["sow_event_date"] == "OFFSET_0":
            h["sow_event_date"] = as_of
    r1 = compute_sow_bear_confirmation(df, holdings, as_of)
    r2 = compute_sow_bear_confirmation(df, holdings, as_of)
    assert r1 == r2


# --- T9: schema de salida (test cubierto en schema test separado) ---
# El test de schema v3 de sector_regime_matrix.csv se anade cuando se
# haga el wiring. Ahora el modulo no toca el CSV.

def test_t9_dict_keys_del_contrato():
    df = _make_market(_market_with_bear_at(299))
    as_of = df.index[-1]
    holdings = _make_holdings(n_valid=6, n_with_sow=3)
    for h in holdings:
        if h["sow_event_date"] == "OFFSET_0":
            h["sow_event_date"] = as_of
    r = compute_sow_bear_confirmation(df, holdings, as_of)
    assert set(r.keys()) == {
        "sow_bear_confirmation", "sow_bear_hits", "sow_bear_reason",
    }
    assert isinstance(r["sow_bear_confirmation"], bool)
    assert isinstance(r["sow_bear_hits"], int)
    assert isinstance(r["sow_bear_reason"], str)


# --- T10: expiracion fuera de ventana M ---

def test_t10_sow_fuera_de_ventana_no_cuenta():
    df = _make_market(_market_with_bear_at(299))
    as_of = df.index[-1]
    # Crear holdings con SOW pero fechas M+3 sesiones atras (fuera)
    sessions = df.index
    old_date = sessions[-M_SOW_WINDOW - 3]
    holdings = _make_holdings(n_valid=6, n_with_sow=3, include_event_date=False)
    for h in holdings:
        if h["sow_status"] == "CONFIRMED":
            h["sow_event_date"] = old_date
    r = compute_sow_bear_confirmation(df, holdings, as_of)
    assert r["sow_bear_confirmation"] is False
    assert r["sow_bear_reason"] == "INSUFFICIENT_SOW_HITS"
    assert r["sow_bear_hits"] == 0


# --- Extras de comportamiento del contrato (no enumerados) ---

def test_is_bear_fail_closed_sin_historico():
    # Menos de MA200 barras: False
    df = _make_market(np.linspace(100, 90, 50))
    as_of = df.index[-1]
    assert is_bear(df[("Close", "SPY")], as_of) is False


def test_count_sow_in_window_vacio():
    sessions = pd.date_range("2024-01-01", periods=10, freq="B")
    assert count_sow_in_window([], sessions, sessions[-1]) == 0
    assert count_sow_in_window([sessions[0]], sessions, sessions[-1]) == 0
    assert count_sow_in_window([sessions[-1]], sessions, sessions[-1]) == 1
