# -*- coding: utf-8 -*-
"""Tests de regresion para el fix C1 en _delta (sector_breadth_momentum.py).

Contexto: _delta rechazaba gaps de calendario superiores a
max(days+2, days*1.6+3). Para days=1 ese limite era 4 dias, insuficiente
para absorber un fin de semana + festivo (caso real 2026-09-18 Vie ->
2026-09-23 Mie = 5 dias). Resultado: delta_1d_ema20 = NaN en 11/11 sectores.
"""

import pandas as pd
import pytest

from indicators.sector_breadth_momentum import _delta


def test_delta_1d_gap_fin_de_semana_mas_festivo():
    """days=1 con gap de 5 dias naturales (Vie->Mie) debe devolver valor."""
    dates = pd.Series(pd.to_datetime([
        "2026-09-16", "2026-09-17", "2026-09-18", "2026-09-23",
    ]))
    series = pd.Series([40.0, 45.0, 60.0, 75.0])
    result = _delta(series, dates, 1)
    assert result == pytest.approx(15.0)  # 75.0 - 60.0


def test_delta_1d_gap_absurdo_sigue_dando_nan():
    """Guard intacto: gap de meses sigue devolviendo NaN."""
    dates = pd.Series(pd.to_datetime(["2026-01-02", "2026-09-23"]))
    series = pd.Series([50.0, 75.0])
    result = _delta(series, dates, 1)
    assert pd.isna(result)


def test_delta_1d_serie_corta_sigue_dando_nan():
    """Guard intacto: serie con menos filas que days+1 -> NaN."""
    dates = pd.Series(pd.to_datetime(["2026-09-23"]))
    series = pd.Series([75.0])
    result = _delta(series, dates, 1)
    assert pd.isna(result)


def test_delta_1d_valores_nan_sigue_dando_nan():
    """Guard intacto: NaN en penultima fila -> NaN."""
    dates = pd.Series(pd.to_datetime([
        "2026-09-18", "2026-09-22", "2026-09-23",
    ]))
    series = pd.Series([60.0, float("nan"), 75.0])
    result = _delta(series, dates, 1)
    assert pd.isna(result)


def test_delta_5d_sin_cambio():
    """days=5 con gap normal sigue funcionando (no regresion)."""
    dates = pd.Series([pd.Timestamp("2026-09-16") + pd.Timedelta(days=i)
                       for i in range(10)])
    series = pd.Series([10.0 * i for i in range(10)])  # 0,10,...,90
    result = _delta(series, dates, 5)
    assert result == pytest.approx(50.0)  # 90 - 40