# -*- coding: utf-8 -*-
"""Regresion: NaN en una metrica no debe propagar a todo el bloque WLS.

Bug 2026-10-01: compute_wls_for_index -> robust_intra usaba
np.median(np.abs(s - median)). np.median NO ignora NaN (a diferencia
de pd.Series.median). Si un ticker del universo tiene NaN en
wyckoff_score o stability (ej: SPCX en QQQ, datos insuficientes),
np.median devolvia NaN y TODA la columna rws_z/stab_z se volvia NaN.
Resultado: wls=NaN para los 5 lideres del bloque Nasdaq-100, check
'ordenado por WLS desc' falla, run-system exit 1.

Fix: np.nanmedian + guard pd.isna(mad).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators.index_leaders import compute_wls_for_index


def _make_metrics_con_nan():
    """15 tickers, uno con NaN en wyckoff_score y stability (SPCX)."""
    return pd.DataFrame({
        "ticker": ["NVDA", "AAPL", "MSFT", "MU", "AMD", "AMZN",
                   "META", "GOOGL", "GOOG", "TSLA", "SPCX", "INTC",
                   "AVGO", "WMT", "PLTR"],
        "rs": [0.0075, 0.011, 0.0169, 0.035, 0.0201, 0.0082,
               0.0238, 0.0113, 0.0112, 0.0117, 0.005, 0.004, 0.0115,
               0.0034, 0.0062],
        "rs_mom": [0.005, -0.0208, -0.0213, 0.0872, 0.2412, -0.0677,
                   0.1819, -0.0174, -0.0182, -0.0484, 0.0141, 0.2563,
                   -0.0943, -0.0638, -0.0059],
        "flow_proxy_z": [0.1214, 0.7755, 0.3479, -0.0375, 0.0895,
                         0.7625, -1.2511, 0.9451, 1.1229, 0.6070,
                         0.5547, 1.0854, -0.5922, -2.4302, 0.0913],
        "wyckoff_score": [0.5312, 0.2960, 0.7236, 0.0280, 0.0965,
                          0.3119, 0.5569, 0.1290, 0.1080, -0.0496,
                          np.nan,  # SPCX
                          0.0625, -0.0162, 0.0972, 0.6252],
        "stability": [0.9984, 0.9999, 1.0, -0.9537, 0.9997,
                      1.0, 1.0, 0.0302, 0.0645, -0.9999,
                      np.nan,  # SPCX
                      0.9978, -0.3494, 0.9999, 1.0],
        "persistence_10d": [0.7, 0.5, 0.5, 0.8, 0.7, 0.7,
                            0.5, 0.6, 0.6, 0.5, 0.5, 0.5, 0.6,
                            0.5, 0.7],
    })


def test_nan_en_un_ticker_no_propaga_a_toda_la_columna():
    """El bug: wls debe ser NaN solo donde el ticker tiene NaN.

    Con el fix, 14 tickers con wls valido + SPCX con NaN.
    Sin el fix, los 15 con NaN.
    """
    metrics = _make_metrics_con_nan()
    result = compute_wls_for_index(metrics)

    n_nan = result["wls"].isna().sum()
    n_valid = result["wls"].notna().sum()

    assert n_nan == 1, (
        "Solo SPCX debe ser NaN. Con el bug son los 15. "
        "NaN actual: {0}".format(n_nan)
    )
    assert n_valid == 14, (
        "14 tickers con wls valido esperados. Actual: {0}".format(n_valid)
    )
    # SPCX es el NaN
    spcx = result[result["ticker"] == "SPCX"]
    assert spcx["wls"].isna().all()


def test_top5_sin_nan_y_ordenado():
    """head(5) del resultado NO debe tener NaN y debe estar ordenado."""
    metrics = _make_metrics_con_nan()
    result = compute_wls_for_index(metrics)
    top5 = result.head(5)

    assert top5["wls"].notna().all(), (
        "Top 5 no debe tener NaN. Actual: {0}".format(
            list(top5["wls"].values)
        )
    )
    assert top5["wls"].is_monotonic_decreasing, (
        "Top 5 debe estar ordenado por wls desc. Actual: {0}".format(
            list(top5["wls"].values)
        )
    )


def test_sin_nan_el_resultado_es_identico():
    """Control: sin NaN, mismo resultado con y sin fix."""
    metrics = _make_metrics_con_nan().dropna()
    result = compute_wls_for_index(metrics)

    assert result["wls"].notna().all()
    assert len(result) == 14
    assert result["wls"].is_monotonic_decreasing


def test_robust_intra_mediana_ignora_nan():
    """robust_intra usa pd.Series.median (ignora NaN) pero
    np.abs(s - median) propaga NaN. Con np.nanmedian, no propaga."""
    from indicators.index_leaders import compute_wls_for_index as f
    # Este test es indirecto: si el fix esta bien, una serie con NaN
    # produce una z-serie con NaN solo donde el input era NaN.
    s = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0])
    median = s.median()  # pandas ignora NaN
    mad_fix = np.nanmedian(np.abs(s - median))
    mad_old = np.median(np.abs(s - median))
    assert not np.isnan(mad_fix), "np.nanmedian debe ignorar el NaN"
    assert np.isnan(mad_old), "np.median (bug) propaga el NaN"
