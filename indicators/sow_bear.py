# -*- coding: utf-8 -*-
"""SOW-BEAR-CONFIRMATION (contrato v1.10.1, doc 57).

Etiqueta informativa por sector. Coexiste con wyckoff_phase.
NO es fase. NO activa DISTRIBUTION. No modifica rankings.

Funciones puras. Sin efectos secundarios. Sin acceso a estado global.
El wiring a sector_regime.py y al CSV espera firma del contrato por
el auditor externo (docs/auditoria/wyckoff/57_contrato_v1_10_1.md,
seccion 9). Este modulo puede existir sin wiring.
"""
from __future__ import annotations

import pandas as pd


SOW_BEAR_MARKET_PROXY = "SPY"
MA200_WINDOW = 200
DRAWDOWN_WINDOW = 252
DRAWDOWN_THRESHOLD = -0.10
M_SOW_WINDOW = 5
N_SOW_MIN_HITS = 3
MIN_VALID_HOLDINGS = 5


def is_bear(close_series, as_of) -> bool:
    """SPY en bear: close < MA200 AND drawdown_252d < -0.10.

    Args:
        close_series: pd.Series con Close de SPY, index temporal.
        as_of: fecha de observacion (Timestamp).

    Returns:
        True si ambas condiciones se cumplen. False si falta historico
        suficiente o no se cumple. Fail-closed (no imputa).
    """
    if close_series is None:
        return False
    sub = close_series.loc[:as_of].dropna()
    if len(sub) < MA200_WINDOW:
        return False
    ma200 = sub.rolling(MA200_WINDOW, min_periods=MA200_WINDOW).mean()
    if pd.isna(ma200.iloc[-1]):
        return False
    dd_window = sub.tail(DRAWDOWN_WINDOW)
    if len(dd_window) == 0:
        return False
    peak = dd_window.max()
    if peak <= 0:
        return False
    last_close = sub.iloc[-1]
    dd = last_close / peak - 1.0
    return bool(last_close < ma200.iloc[-1] and dd < DRAWDOWN_THRESHOLD)


def count_sow_in_window(sow_dates, sessions, as_of, m=M_SOW_WINDOW) -> int:
    """Cuenta fechas SOW en [as_of - m + 1 sesiones, as_of].

    Args:
        sow_dates: lista de Timestamps (event_dates de holdings con SOW).
        sessions: DatetimeIndex con las sesiones del universo.
        as_of: fecha de observacion.
        m: ventana en sesiones (default M_SOW_WINDOW=5).

    Returns:
        Numero de sow_dates dentro de la ventana (inclusive).
    """
    if not sow_dates:
        return 0
    sessions_valid = sessions[sessions <= as_of]
    if len(sessions_valid) < m:
        return 0
    window_start = sessions_valid[-m]
    window_end = sessions_valid[-1]
    n = 0
    for d in sow_dates:
        if d is None:
            continue
        try:
            ts = pd.Timestamp(d)
        except (ValueError, TypeError):
            continue
        if window_start <= ts <= window_end:
            n += 1
    return n


def compute_sow_bear_confirmation(
    df_market,
    holdings_sector,
    as_of,
    market_proxy=SOW_BEAR_MARKET_PROXY,
) -> dict:
    """Etiqueta SOW_BEAR_CONFIRMATION por sector.

    Args:
        df_market: DataFrame MultiIndex con columnas (field, ticker).
        holdings_sector: lista de dicts, uno por holding del sector:
            {ticker, sow_status, sow_event_date, evaluable}.
        as_of: fecha de observacion.
        market_proxy: ticker del proxy (default SPY).

    Returns:
        dict con keys:
            sow_bear_confirmation (bool)
            sow_bear_hits (int)
            sow_bear_reason (str): OK | NOT_BEAR |
                INSUFFICIENT_SOW_HITS | INSUFFICIENT_COVERAGE |
                NO_DATA_MARKET_PROXY | NO_DATA_SECTOR
    """
    # Fail-closed: SPY ausente
    if df_market is None or ("Close", market_proxy) not in df_market.columns:
        return {
            "sow_bear_confirmation": False,
            "sow_bear_hits": 0,
            "sow_bear_reason": "NO_DATA_MARKET_PROXY",
        }

    if not holdings_sector:
        return {
            "sow_bear_confirmation": False,
            "sow_bear_hits": 0,
            "sow_bear_reason": "NO_DATA_SECTOR",
        }

    sessions = df_market.index
    spy_close = df_market[("Close", market_proxy)]

    if not is_bear(spy_close, as_of):
        return {
            "sow_bear_confirmation": False,
            "sow_bear_hits": 0,
            "sow_bear_reason": "NOT_BEAR",
        }

    # Contar holdings con SOW en ventana
    n_valid = 0
    sow_dates_in_window = []
    for h in holdings_sector:
        if not h.get("evaluable", False):
            continue
        n_valid += 1
        if h.get("sow_status") != "CONFIRMED":
            continue
        event_date = h.get("sow_event_date")
        if event_date is None:
            continue
        if count_sow_in_window([event_date], sessions, as_of) > 0:
            sow_dates_in_window.append(event_date)

    n_hits = len(sow_dates_in_window)

    if n_valid < MIN_VALID_HOLDINGS:
        return {
            "sow_bear_confirmation": False,
            "sow_bear_hits": n_hits,
            "sow_bear_reason": "INSUFFICIENT_COVERAGE",
        }

    if n_hits < N_SOW_MIN_HITS:
        return {
            "sow_bear_confirmation": False,
            "sow_bear_hits": n_hits,
            "sow_bear_reason": "INSUFFICIENT_SOW_HITS",
        }

    return {
        "sow_bear_confirmation": True,
        "sow_bear_hits": n_hits,
        "sow_bear_reason": "OK",
    }
