# -*- coding: utf-8 -*-
"""REP-AUDIT-20261017-H7: _resolve_breadth_effective_date NO debe contar
ETFs sectoriales de df_market en el denominador de cobertura temporal.

Bug detectado en el run del 2026-10-17 21:11. Los ETFs sectoriales se
anadian a `eligible` "como input de RS", pero resolve_effective_date
opera sobre df_stocks. Los ETFs no estan en df_stocks.columns, asi que
se contaban como "no observados" en el denominador sin poder sumar en
el numerador. Resultado: 305/340=0.897 < 0.90, retroceso al 15,
DATA_PENDING falso. Con el universo correcto: 305/329=0.927 >= 0.90.
"""
import numpy as np
import pandas as pd

from src.pipeline.breadth_metrics import _resolve_breadth_effective_date


def _make_stocks(n_tickers, dates, cov_by_date):
    """DataFrame MultiIndex (Close, Ticker) x fechas.

    Construye una matrix numpy y luego el DataFrame, para evitar
    ambiguedades de asignacion con MultiIndex columns.
    """
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    data = np.full((len(dates), n_tickers), np.nan, dtype=float)
    for j, d_iso in enumerate(dates):
        if d_iso in cov_by_date:
            n_valid = cov_by_date[d_iso]
            data[j, :n_valid] = 100.0
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    cols = pd.MultiIndex.from_product([["Close"], tickers])
    return pd.DataFrame(data, index=index, columns=cols)


def _make_market(n_etfs, dates):
    """DataFrame MultiIndex (Close, ETF) x fechas. Todos con Close."""
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    if n_etfs <= 0:
        return pd.DataFrame(index=index)
    etfs = [f"XL{i:02d}" for i in range(n_etfs)]
    data = np.full((len(dates), n_etfs), 100.0, dtype=float)
    cols = pd.MultiIndex.from_product([["Close"], etfs])
    return pd.DataFrame(data, index=index, columns=cols)


def _make_holdings(etfs):
    if not etfs:
        return pd.DataFrame({"etf": [], "ticker": [], "weight": []})
    return pd.DataFrame({
        "etf": etfs,
        "ticker": [f"C{i}" for i in range(len(etfs))],
        "weight": [1.0 / len(etfs)] * len(etfs),
    })


def test_h7_etfs_no_inflan_denominador():
    """329 equities, 305 con Close el 16 (305/329=0.927 >= 0.90).

    Antes del fix: 305/340 = 0.897 < 0.90 -> effective_date = 15.
    Con el fix: 305/329 = 0.927 >= 0.90 -> effective_date = 16.
    """
    n_eq = 329
    dates = ["2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16"]
    stocks = _make_stocks(n_eq, dates, {
        "2026-10-13": 320,
        "2026-10-14": 320,
        "2026-10-15": 320,
        "2026-10-16": 305,
    })
    etfs = [f"XL{i:02d}" for i in range(11)]
    market = _make_market(11, dates)
    holdings = _make_holdings(etfs)

    eff_date, meta = _resolve_breadth_effective_date(
        stocks, market, holdings, min_coverage=0.90,
    )
    assert eff_date is not None
    assert pd.Timestamp(eff_date).date() == pd.Timestamp("2026-10-16").date(), (
        "Con fix H7, 305/329=0.927 debe dar 16. Got {} (cov={})".format(
            eff_date, meta.get("coverage"),
        )
    )
    assert meta["n_eligible"] == n_eq
    assert meta["coverage"] >= 0.90


def test_h7_sin_etfs_el_comportamiento_no_cambia():
    """Sin ETFs sectoriales, el calculo es identico al anterior."""
    dates = ["2026-10-13", "2026-10-14", "2026-10-15", "2026-10-16"]
    stocks = _make_stocks(100, dates, {
        "2026-10-13": 95, "2026-10-14": 95,
        "2026-10-15": 95, "2026-10-16": 90,
    })
    market = _make_market(0, dates)
    holdings = _make_holdings([])

    eff_date, meta = _resolve_breadth_effective_date(
        stocks, market, holdings, min_coverage=0.90,
    )
    assert eff_date is not None
    assert pd.Timestamp(eff_date).date() == pd.Timestamp("2026-10-16").date()
    assert meta["n_eligible"] == 100


def test_h7_etfs_no_estan_en_stocks_no_cuentan():
    """ETFs en df_market pero no en df_stocks: no se cuentan."""
    dates = ["2026-10-15", "2026-10-16"]
    stocks = _make_stocks(10, dates, {
        "2026-10-15": 10, "2026-10-16": 9,
    })
    etfs = ["XLB", "XLE", "XLF"]
    market = _make_market(len(etfs), dates)
    holdings = _make_holdings(etfs)

    eff_date, meta = _resolve_breadth_effective_date(
        stocks, market, holdings, min_coverage=0.90,
    )
    assert meta["n_eligible"] == 10
    assert pd.Timestamp(eff_date).date() == pd.Timestamp("2026-10-16").date()


def test_h7_cobertura_justo_en_umbral():
    """296/329 = 0.8997 < 0.90, retrocede al 15."""
    dates = ["2026-10-15", "2026-10-16"]
    stocks = _make_stocks(329, dates, {
        "2026-10-15": 320,
        "2026-10-16": 296,
    })
    market = _make_market(11, dates)
    holdings = _make_holdings([f"XL{i:02d}" for i in range(11)])

    eff_date, meta = _resolve_breadth_effective_date(
        stocks, market, holdings, min_coverage=0.90,
    )
    assert pd.Timestamp(eff_date).date() == pd.Timestamp("2026-10-15").date()
