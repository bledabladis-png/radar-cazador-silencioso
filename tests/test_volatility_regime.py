"""Tests C-08 (2026-09-29): compute_volatility_regime y NaN.

Bug detectado en auditoria funcional del bloque C: NaN en el ultimo z
caia silenciosamente a STRESS (todas las comparaciones con NaN dan
False -> else). 1803/1884 detecciones historicas de STRESS eran falsos
positivos por NaN heredado de huecos de VIX.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimes.volatility_regime import compute_volatility_regime


def test_c08_serie_vacia_devuelve_nd():
    z, regime, conf = compute_volatility_regime(pd.Series([], dtype=float))
    assert regime == "N/D"
    assert conf == 0.0


def test_c08_ultimo_nan_devuelve_nd():
    z, regime, conf = compute_volatility_regime(pd.Series([np.nan]))
    assert regime == "N/D"
    assert conf == 0.0


def test_c08_todo_nan_devuelve_nd():
    z, regime, conf = compute_volatility_regime(pd.Series([np.nan] * 100))
    assert regime == "N/D"
    assert conf == 0.0
