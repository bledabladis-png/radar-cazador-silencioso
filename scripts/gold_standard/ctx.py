# -*- coding: utf-8 -*-
"""Variable contextual (ctx) para el diseno v8.

Protocolo v8, seccion 3.

    ctx_t = 1  si y solo si:
      (a) MA50(t-60) > MA200(t-60)
      AND
      (b) (max(Close,[t-59,t]) - Close_t) / max(Close,[t-59,t]) >= 0.15

- ctx se calcula EXCLUSIVAMENTE sobre OHLCV.
- No importa wyckoff_v1, detect_sow, config ni ningun otro modulo.
- ctx NO genera Y ni modifica el detector.
- ctx NO se muestra al anotador.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import pandas as pd

MA_FAST = 50
MA_SLOW = 200
MA_LOOKBACK = 60         # (a) se evalua con MA en t-60
DD_WINDOW = 60
DD_UMBRAL = 0.15


def compute_ctx(
    df_ticker: pd.DataFrame,
    ma_fast: int = MA_FAST,
    ma_slow: int = MA_SLOW,
    ma_lookback: int = MA_LOOKBACK,
    dd_window: int = DD_WINDOW,
    dd_umbral: float = DD_UMBRAL,
) -> pd.Series:
    """Serie booleana de ctx por fecha.

    df_ticker: OHLCV flat con columnas Open/High/Low/Close/Volume.
               Debe ser el df limpio (build_ticker_df), sin NaN.

    Devuelve Serie booleana indexada por fecha.
    """
    for c in ("Close",):
        if c not in df_ticker.columns:
            raise ValueError(f"Falta columna {c}")

    close = df_ticker["Close"]

    # (a) MA50(t-60) > MA200(t-60)
    ma50 = close.rolling(ma_fast, min_periods=ma_fast).mean()
    ma200 = close.rolling(ma_slow, min_periods=ma_slow).mean()
    a_series = (ma50 > ma200).shift(ma_lookback)

    # (b) (max(Close,[t-59,t]) - Close_t) / max(Close,[t-59,t]) >= umbral
    rolling_max = close.rolling(dd_window, min_periods=dd_window).max()
    dd_series = (rolling_max - close) / rolling_max
    b_series = dd_series >= dd_umbral

    # (a_series == True) convierte NaN a False sin downcasting
    a_bool = (a_series == True)  # noqa: E712
    b_bool = (b_series == True)  # noqa: E712
    ctx = a_bool & b_bool
    ctx.name = "ctx"
    return ctx


def ctx_stats(ctx: pd.Series) -> dict:
    """Resumen descriptivo."""
    n = int(len(ctx))
    n_pos = int(ctx.sum())
    return {
        "n_total": n,
        "n_ctx1": n_pos,
        "n_ctx0": n - n_pos,
        "pct_ctx1": round(100.0 * n_pos / n, 3) if n else 0.0,
    }