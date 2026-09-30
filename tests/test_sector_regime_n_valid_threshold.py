# -*- coding: utf-8 -*-
"""H1: score = NaN si n_valid < 4.

Contrato (decision de diseno 2026-09-30):
  n_valid < 4  -> score = NaN
  n_valid >= 4 -> score calculable

Razon: con 4 componentes validos, el peso minimo representado es
0.10 + 0.15 + 0.15 + 0.15 = 0.55. Con 3 o menos, la renormalizacion
convierte una fraccion arbitrariamente pequena en el 100% del score.

Aplicable al warm-up inicial y a XLC pre-2018.
"""
import numpy as np
import pandas as pd

from config.tickers import MARKET_TICKERS
from regimes.sector_regime import compute_sector_scores


def _make_df_one_sector_short():
    """300 sesiones para los 10 sectores regulares, XLE solo ultimas 15.

    Resultado esperado en la ultima fecha:
      - XLE: n_valid=2 (solo trend y breadth; rs_mom_* y vol_inv
        en warm-up) -> NaN por umbral n_valid>=4
      - Resto: n_valid=6 -> score normal

    Nota: 60 sesiones dan n_valid=4 (rs_mom_20, trend, vol_inv,
    breadth). Justo en el umbral. Para forzar <4 hace falta 15.
    """
    sectors = MARKET_TICKERS["sectors"]
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.RandomState(42)
    data = {}
    data[("Close", "^GSPC")] = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    sorted_sectors = sorted(sectors)
    for s in sorted_sectors:
        k = sorted_sectors.index(s)
        drift = 0.0001 * (k + 1)
        close = 100 * np.cumprod(1 + rng.normal(drift, 0.015, n))
        if s == "XLE":
            close = close.copy()
            close[: n - 15] = np.nan  # solo ultimas 15 sesiones
        data[("Close", s)] = close
        data[("Open", s)] = close * 0.998
        data[("High", s)] = close * 1.01
        data[("Low", s)] = close * 0.99
        data[("Volume", s)] = rng.randint(1000000, 5000000, n).astype(float)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_sector_con_n_valid_bajo_es_nan():
    """XLE con close solo en 60 sesiones no puede publicar score."""
    df = _make_df_one_sector_short()
    res = compute_sector_scores(df)
    assert res is not None
    last = res["last_scores"]
    assert pd.isna(last.get("XLE")), (
        f"XLE deberia ser NaN (n_valid<4), es {last.get('XLE')}"
    )


def test_sector_con_n_valid_alto_publica_score():
    """XLK con 300 sesiones publica score normal."""
    df = _make_df_one_sector_short()
    res = compute_sector_scores(df)
    assert res is not None
    last = res["last_scores"]
    assert pd.notna(last.get("XLK")), (
        f"XLK deberia tener score, es {last.get('XLK')}"
    )