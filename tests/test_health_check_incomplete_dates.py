# -*- coding: utf-8 -*-
"""Tests Commit B (2026-09-24): CONFIRMED_INCOMPLETE_DATES.

Contexto: el 22-Sep tiene 34% de cobertura (fallo puntual de Yahoo,
206/313 tickers USA sin Close). El health check lo marcaba como WARN
en cada ejecucion. Se marca como fecha historicamente incompleta y se
excluye SOLO de check_coverage_last_5.
"""

import pandas as pd

from scripts.health_check import (
    CONFIRMED_INCOMPLETE_DATES,
    OK,
    check_coverage_last_5,
)


def _make_df(dates, coverage_by_date):
    """coverage_by_date: {fecha: n_validos} para 3 tickers.

    None = todos los tickers con Close.
    """
    data = {}
    for t in ["T1", "T2", "T3"]:
        data[("Close", t)] = [100.0] * len(dates)
    df = pd.DataFrame(data, index=pd.DatetimeIndex(dates))
    # Aplicar cobertura parcial
    for fecha_str, n_valid in coverage_by_date.items():
        if n_valid is None:
            continue
        fecha = pd.Timestamp(fecha_str)
        if fecha not in df.index:
            continue
        for i, t in enumerate(["T1", "T2", "T3"]):
            if i >= n_valid:
                df.loc[fecha, ("Close", t)] = float("nan")
    return df


def test_b_constante_contiene_22sep():
    """La constante debe incluir el 22-Sep."""
    assert "2026-09-22" in CONFIRMED_INCOMPLETE_DATES


def test_b_22sep_no_genera_warn_hist():
    """Con 22-Sep al 34% y marcado como incompleto, no debe salir WARN."""
    dates = ["2026-09-17", "2026-09-18", "2026-09-21",
             "2026-09-22", "2026-09-23"]
    df = _make_df(dates, {"2026-09-22": 1, "2026-09-23": None})
    results = check_coverage_last_5(df)
    hist = [r for r in results if r.name == "coverage:hist"][0]
    assert hist.status == OK
    assert "excluidas" in hist.message