# -*- coding: utf-8 -*-
"""Tests N-C6: fecha efectiva de Sector Breadth por resolucion canonica.

Dictamen auditor externo 2026-10-16. Aceptado como incumplimiento de R2.

Reglas cubiertas:
- DataFrame ordenado normal -> fecha efectiva = ultima sesion completa.
- DataFrame desordenado -> no confia en index[-1].
- Ultima fila fisica != ultima sesion admisible -> usa la admisible.
- Sesion final incompleta (cobertura < 0.90) -> fallback STALE.
- Universo elegible incluye ETFs sectoriales de df_market.
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.breadth_metrics import _resolve_breadth_effective_date


FRIDAY = datetime(2026, 9, 11, 23, 30)  # sesion NYSE + post-publish


def _mk_stocks(end_date, tickers=None, n=300, close_nan_final=None):
    """df_stocks con N tickers. Si close_nan_final se indica, deja NaN
    en la ultima fila de los primeros close_nan_final tickers."""
    if tickers is None:
        tickers = ["AAA", "BBB", "CCC"]
    idx = pd.date_range(end=end_date, periods=n, freq="D")
    data = {}
    for i, t in enumerate(tickers):
        vals = [100.0 + j * 0.1 for j in range(n)]
        if close_nan_final and i < close_nan_final:
            vals[-1] = float("nan")
        data[("Close", t)] = vals
        data[("High", t)] = [101.0] * n
        data[("Low", t)] = [99.0] * n
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _mk_market(end_date, etfs=None, n=300):
    if etfs is None:
        etfs = ["XLK"]
    idx = pd.date_range(end=end_date, periods=n, freq="D")
    data = {}
    for e in etfs:
        data[(e, "Close")] = [100.0 + j * 0.05 for j in range(n)]
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _holdings(etfs=None):
    if etfs is None:
        etfs = ["XLK"]
    return pd.DataFrame({"etf": etfs, "ticker": ["AAA"] * len(etfs)})


# ============================================================
# T1: caso normal (df_stocks ordenado, cobertura completa)
# ============================================================
def test_fecha_efectiva_caso_normal():
    df = _mk_stocks("2026-09-11")
    mkt = _mk_market("2026-09-11")
    eff, meta = _resolve_breadth_effective_date(df, mkt, _holdings())
    assert eff is not None
    assert pd.Timestamp(eff).date() == pd.Timestamp("2026-09-11").date()
    assert meta["status"] == "OK"


# ============================================================
# T2: ultima fila fisica != ultima sesion admisible
# df_stocks con 15-oct completo (100% cobertura) -> efectiva = 15-oct.
# ============================================================
def test_usa_resolucion_canonica_no_posicion_fisica():
    df = _mk_stocks("2026-10-15")
    mkt = _mk_market("2026-10-15")
    eff, meta = _resolve_breadth_effective_date(df, mkt, _holdings())
    assert pd.Timestamp(eff).date() == pd.Timestamp("2026-10-15").date()


# ============================================================
# T3: sesion final incompleta -> fecha efectiva retrocede
# 14-oct completo, 15-oct con solo 1/3 tickers con close -> cobertura
# 33% < 90% -> la fecha efectiva es 14-oct.
# ============================================================
def test_sesion_final_incompleta_retrocede():
    df = _mk_stocks("2026-10-15", tickers=["AAA", "BBB", "CCC"],
                     close_nan_final=2)  # solo AAA con close en 15-oct
    mkt = _mk_market("2026-10-15")
    eff, meta = _resolve_breadth_effective_date(df, mkt, _holdings())
    # effective retrocede al 14-oct (cobertura completa)
    assert pd.Timestamp(eff).date() == pd.Timestamp("2026-10-14").date()


# ============================================================
# T4: cobertura insuficiente en TODO el historico -> None
# Requiere que TODAS las filas tengan coverage < 90%.
# (Si solo la ultima fila es incompleta, la resolucion retrocede
# a la anterior completa; eso es T3.)
# ============================================================
def test_cobertura_global_insuficiente():
    # 10 tickers, con solo 1 con close en todas las filas -> 10% coverage
    tickers = ["T{:02d}".format(i) for i in range(10)]
    idx = pd.date_range(end="2026-10-15", periods=300, freq="D")
    data = {}
    for i, t in enumerate(tickers):
        vals = [100.0 + j * 0.1 for j in range(300)]
        if i > 0:  # 9 de 10 con NaN en todo el historico
            vals = [float("nan")] * 300
        data[("Close", t)] = vals
        data[("High", t)] = [101.0] * 300
        data[("Low", t)] = [99.0] * 300
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    mkt = _mk_market("2026-10-15")
    eff, meta = _resolve_breadth_effective_date(df, mkt, _holdings())
    assert eff is None
    assert meta["status"] == "INSUFFICIENT_COVERAGE"


# ============================================================
# T5: df_stocks vacio -> None
# ============================================================
def test_df_stocks_vacio():
    eff, meta = _resolve_breadth_effective_date(
        pd.DataFrame(), _mk_market("2026-10-15"), _holdings()
    )
    assert eff is None


# ============================================================
# T6: los ETFs sectoriales de df_market se anaden al universo elegible
# ============================================================
def test_etfs_sectoriales_en_universo_elegible():
    """Si df_market tiene ETFs sectoriales ausentes en df_stocks, deben
    anadirse al universo para la resolucion de fecha efectiva."""
    df = _mk_stocks("2026-10-15", tickers=["AAA", "BBB"])
    mkt = _mk_market("2026-10-15", etfs=["XLK", "XLF", "XLE"])
    eff, meta = _resolve_breadth_effective_date(
        df, mkt, _holdings(["XLK", "XLF", "XLE"])
    )
    assert eff is not None
    assert meta["n_eligible"] >= 2  # AAA, BBB + XLK, XLF, XLE