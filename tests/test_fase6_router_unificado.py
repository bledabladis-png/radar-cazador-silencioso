# -*- coding: utf-8 -*-
"""Fase 6 (2026-10-26): router unificado para fed_data.

Cubre:
- router.get_fed_data(columns=[...]) filtra columnas y hace dropna(how='all').
- columns inexistentes -> None.
- regimes._fetch_liquidity_v2_data delega en DataRouter en vez de
  duplicar la politica CSV-first.
"""
import pandas as pd

from data.providers.router import DataRouter
from src.pipeline import regimes


def _df_full():
    idx = pd.date_range("2026-10-01", periods=3)
    return pd.DataFrame({
        "sofr": [1.5, 1.6, 1.7],
        "effr": [1.4, 1.5, 1.6],
        "fed_balance": [8000.0, 8010.0, 8020.0],
        "iorb": [1.3, 1.4, 1.5],
    }, index=idx)


def test_router_get_fed_data_con_columns_filtra(monkeypatch):
    """columns=[sofr,effr] devuelve solo esas columnas, sin otras."""
    router = DataRouter()
    monkeypatch.setattr("data.providers.router.os.path.exists", lambda p: True)
    monkeypatch.setattr(router, "_load_macro_manual", lambda d: _df_full())

    out = router.get_fed_data(columns=["sofr", "effr"])
    assert out is not None
    assert list(out.columns) == ["sofr", "effr"]
    assert len(out) == 3


def test_router_get_fed_data_columns_inexistentes_devuelve_none(monkeypatch):
    """columns que no estan en el df -> None."""
    router = DataRouter()
    monkeypatch.setattr("data.providers.router.os.path.exists", lambda p: True)
    monkeypatch.setattr(router, "_load_macro_manual", lambda d: _df_full())

    out = router.get_fed_data(columns=["nonexistent"])
    assert out is None


def test_router_get_fed_data_sin_columns_devuelve_todo(monkeypatch):
    """Compat: sin columns, comportamiento anterior (todas las target)."""
    router = DataRouter()
    monkeypatch.setattr("data.providers.router.os.path.exists", lambda p: True)
    monkeypatch.setattr(router, "_load_macro_manual", lambda d: _df_full())

    out = router.get_fed_data()
    assert out is not None
    assert set(out.columns) == {"sofr", "effr", "fed_balance", "iorb"}


def test_regimes_usa_router_get_fed_data(monkeypatch):
    """_fetch_liquidity_v2_data delega en DataRouter.get_fed_data(columns=[sofr,effr])."""
    calls = {"n": 0, "kwargs": None}

    class FakeRouter:
        def get_fed_data(self, columns=None):
            calls["n"] += 1
            calls["kwargs"] = {"columns": columns}
            return pd.DataFrame({
                "sofr": [1.5, 1.6],
                "effr": [1.4, 1.5],
            }, index=pd.date_range("2026-10-01", periods=2))

    monkeypatch.setattr("data.providers.router.DataRouter", FakeRouter)

    # Evitar red para p99 NY Fed.
    import requests
    def _fake_get(*a, **kw):
        raise RuntimeError("test: no network")
    monkeypatch.setattr(requests, "get", _fake_get)

    sofr, effr, c1 = regimes._fetch_liquidity_v2_data()

    assert calls["n"] == 1
    assert calls["kwargs"]["columns"] == ["sofr", "effr"]
    assert sofr is not None and len(sofr) == 2
    assert effr is not None and len(effr) == 2
    assert c1 is None


def test_regimes_router_devuelve_none_sin_datos(monkeypatch):
    """Si el router no trae datos, _fetch_liquidity_v2_data -> (None,None,None)."""
    class FakeRouter:
        def get_fed_data(self, columns=None):
            return None

    monkeypatch.setattr("data.providers.router.DataRouter", FakeRouter)
    import requests
    monkeypatch.setattr(requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no net")))

    sofr, effr, c1 = regimes._fetch_liquidity_v2_data()
    assert sofr is None and effr is None and c1 is None
