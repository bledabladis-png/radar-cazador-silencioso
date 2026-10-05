# -*- coding: utf-8 -*-
"""Sampling frame del estudio Gold Standard SOW v1.9.

Determina el conjunto de (ticker, t) elegibles antes del muestreo.

Condiciones de elegibilidad (protocolo v7, seccion 2.1):
  - warmup computacional >= 200 sesiones por ticker
  - observaciones validas hasta t >= 240 (ventana visual completa)
  - OHLCV completo en las 240 sesiones visuales hasta t

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.gold_standard.constants import (
    WARMUP_MIN,
    VISUAL_WINDOW,
    FIELDS_REQUIRED,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PARQUET = ROOT / "data" / "stock_prices.parquet"


@dataclass
class SamplingFrame:
    """Resultado del sampling frame."""
    episodes: pd.DataFrame                # columnas: ticker, t
    n_frame: int
    n_frame_by_ticker: pd.Series
    diagnostics: dict


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Carga stock_prices.parquet. MultiIndex (field, ticker)."""
    p = path if path is not None else DEFAULT_PARQUET
    if not p.exists():
        raise FileNotFoundError(f"No existe {p}")
    return pd.read_parquet(p)


def _validate_columns(df: pd.DataFrame) -> None:
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Se espera MultiIndex (field, ticker) en columnas")
    fields = set(df.columns.get_level_values(0))
    missing = set(FIELDS_REQUIRED) - fields
    if missing:
        raise ValueError(f"Faltan campos OHLCV: {sorted(missing)}")


def _ohlcv_complete_mask(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Serie booleana por fecha: True si OHLCV completo ese dia."""
    sub = df.xs(ticker, axis=1, level=1)
    return sub[list(FIELDS_REQUIRED)].notna().all(axis=1)


def build_sampling_frame(
    df: pd.DataFrame,
    warmup_min: int = WARMUP_MIN,
    visual_window: int = VISUAL_WINDOW,
) -> SamplingFrame:
    """Devuelve los (ticker, t) elegibles.

    Para cada ticker:
      1. Serie booleana de OHLCV completo.
      2. Rolling sum de visual_window; elegible si == visual_window.
      3. Excluir posiciones con ordinal < warmup_min + visual_window - 1.
    """
    _validate_columns(df)
    tickers = sorted(df.columns.get_level_values(1).unique())

    rows = []
    n_by_ticker = {}
    for tk in tickers:
        complete = _ohlcv_complete_mask(df, tk).astype(int)
        if complete.sum() < warmup_min + visual_window:
            n_by_ticker[tk] = 0
            continue
        roll = complete.rolling(
            window=visual_window, min_periods=visual_window
        ).sum()
        mask = (roll == visual_window).to_numpy()
        positions = np.where(mask)[0]
        min_pos = warmup_min + visual_window - 1
        positions = positions[positions >= min_pos]
        dates = df.index[positions]
        for d in dates:
            rows.append({"ticker": tk, "t": d})
        n_by_ticker[tk] = int(len(dates))

    episodes = pd.DataFrame(rows, columns=["ticker", "t"])
    return SamplingFrame(
        episodes=episodes,
        n_frame=len(episodes),
        n_frame_by_ticker=pd.Series(n_by_ticker, name="n_frame"),
        diagnostics={
            "warmup_min": warmup_min,
            "visual_window": visual_window,
            "n_tickers_input": len(tickers),
            "n_tickers_with_frame": int(
                (pd.Series(n_by_ticker) > 0).sum()
            ),
        },
    )