"""Tests unitarios de pipeline_contractual (E.2.d).

Cubren las piezas puras: ticker_of y _subset_universe.
La orquestacion (run_contractual_nipc) se cubre por E2E via
scripts/iae_contractual_nipc_e2e.py (E.1 PASS).
"""
from __future__ import annotations

import pandas as pd

from src.institutional_accumulation.pipeline_contractual import (
    ticker_of,
    _subset_universe,
)
from src.institutional_accumulation.identity.target_builder import (
    TargetUniverse,
)


def _make_universe(keys):
    """TargetUniverse minimo con coherencia ticker/figi/row_uid."""
    keys = list(keys)
    return TargetUniverse(
        version_id="v_test",
        period_end="2026-03-31",
        catalog_version_id="cat_test",
        catalog_sha256="a" * 64,
        declared_keys=set(keys),
        ticker_by_key={k: "TK_" + k for k in keys},
        figi_by_key={k: "FG_" + k for k in keys},
        row_uid_by_key={k: "UID_" + k for k in keys},
        key_by_row_uid={"UID_" + k: k for k in keys},
    )

def test_ticker_of_equity_extrae_ticker():
    s = pd.Series(["equity:AAPL", "equity:MSFT", "equity:GOOGL"])
    out = ticker_of(s)
    assert list(out) == ["AAPL", "MSFT", "GOOGL"]


def test_ticker_of_figi_devuelve_none():
    s = pd.Series(["figi:BBG001S5N8V8", "figi:BBG009S3NB21"])
    out = ticker_of(s)
    assert out.isna().all()


def test_ticker_of_nan_y_vacios():
    s = pd.Series([None, "", "   ", float("nan")])
    out = ticker_of(s)
    assert out.isna().all()


def test_ticker_of_mixto():
    s = pd.Series(["equity:AAPL", "figi:BBG001", None, "cusip:037833100"])
    out = ticker_of(s)
    assert out.iloc[0] == "AAPL"
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[2])
    assert pd.isna(out.iloc[3])


def test_subset_universe_reduce_keys():
    u = _make_universe(["k1", "k2", "k3"])
    u2 = _subset_universe(u, {"k1", "k2"})
    assert u2.declared_keys == {"k1", "k2"}
    assert set(u2.ticker_by_key.keys()) == {"k1", "k2"}
    assert set(u2.figi_by_key.keys()) == {"k1", "k2"}
    assert set(u2.row_uid_by_key.keys()) == {"k1", "k2"}
    assert u2.key_by_row_uid == {"UID_k1": "k1", "UID_k2": "k2"}


def test_subset_universe_preserva_metadata():
    u = _make_universe(["k1", "k2", "k3"])
    u2 = _subset_universe(u, {"k2"})
    assert u2.version_id == u.version_id
    assert u2.period_end == u.period_end
    assert u2.catalog_version_id == u.catalog_version_id
    assert u2.catalog_sha256 == u.catalog_sha256
    assert u2.ticker_by_key == {"k2": "TK_k2"}