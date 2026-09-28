"""Tests de sector_leader_divergence.

F4-06a (2026-09-28): reescritos. Los tests originales eran:
  - test_classification_logic: `assert True` con comentario de placeholder.
  - test_n_valid_less_than_3_returns_nd: reimplementaba la logica de
    clasificacion localmente sin llamar a la funcion real.

Los tests nuevos verifican _ret_20d (funcion pura del modulo).
compute_sector_leader_divergence (funcion principal) queda sin cobertura
directa: requiere df_stocks + holdings_df + leader_df + df_market +
temporal_meta. Deuda declarada.
"""
import pandas as pd
import pytest

from indicators.sector_leader_divergence import _ret_20d


def test_ret_20d_nan_si_menos_de_21_valores():
    close = pd.Series([100.0] * 20)
    assert pd.isna(_ret_20d(close))


def test_ret_20d_calcula_variacion_20_sesiones():
    close = pd.Series([100.0] * 20 + [110.0])
    assert _ret_20d(close) == pytest.approx(0.10)
