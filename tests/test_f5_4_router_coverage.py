"""F5-4 (2026-09-28): router rechaza subset parcial silencioso (A5-15).

Sin red. Se reemplazan los providers por fakes para controlar cobertura
y fallos.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.providers.router import DataRouter


def _make_multiindex_df(tickers, n_rows=10):
    dates = pd.date_range("2026-01-01", periods=n_rows, freq="D")
    cols = pd.MultiIndex.from_product([["Close"], tickers])
    return pd.DataFrame(1.0, index=dates, columns=cols)


class _FakeProvider:
    def __init__(self, name, available=True, prices=None, raises=None):
        self._name = name
        self._available = available
        self._prices = prices
        self._raises = raises
        self.calls = 0

    def get_name(self):
        return self._name

    def is_available(self):
        return self._available

    def get_prices(self, tickers, period="10y", **kw):
        self.calls += 1
        if self._raises:
            raise self._raises
        return self._prices


def _install_fakes(router, providers_dict):
    router.providers = providers_dict


def test_router_rechaza_subset_parcial_y_prueba_siguiente(monkeypatch):
    """A5-15: proveedor con subset parcial no es aceptado; se prueba el siguiente."""
    router = DataRouter()
    requested = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    partial = _make_multiindex_df(["AAPL", "MSFT", "GOOGL"])
    complete = _make_multiindex_df(requested)

    yahoo = _FakeProvider("Yahoo Fake", prices=partial)
    polygon = _FakeProvider("Polygon Fake", prices=complete)
    fred = _FakeProvider("FRED Fake", prices=None)

    _install_fakes(router, {"yahoo": yahoo, "polygon": polygon, "fred": fred})

    out = router.get_market_data(requested, period="10y")
    assert not out.empty
    assert yahoo.calls == 1
    assert polygon.calls == 1
    assert fred.calls == 0
    covered = {c[1] for c in out.columns}
    assert covered == set(requested)


def test_router_acepta_cobertura_completa_sin_probar_siguiente():
    router = DataRouter()
    requested = ["AAPL", "MSFT", "GOOGL"]
    complete = _make_multiindex_df(requested)

    yahoo = _FakeProvider("Yahoo Fake", prices=complete)
    polygon = _FakeProvider("Polygon Fake", prices=complete)
    fred = _FakeProvider("FRED Fake", prices=None)

    _install_fakes(router, {"yahoo": yahoo, "polygon": polygon, "fred": fred})

    out = router.get_market_data(requested, period="10y")
    assert not out.empty
    assert yahoo.calls == 1
    assert polygon.calls == 0
    assert fred.calls == 0


def test_router_excepcion_pasa_a_siguiente():
    router = DataRouter()
    requested = ["AAPL", "MSFT"]
    complete = _make_multiindex_df(requested)

    yahoo = _FakeProvider("Yahoo Fake", raises=RuntimeError("mock error"))
    polygon = _FakeProvider("Polygon Fake", prices=complete)
    fred = _FakeProvider("FRED Fake", prices=None)

    _install_fakes(router, {"yahoo": yahoo, "polygon": polygon, "fred": fred})

    out = router.get_market_data(requested, period="10y")
    assert not out.empty
    assert yahoo.calls == 1
    assert polygon.calls == 1
    assert fred.calls == 0


def test_router_fallback_a_cache_si_todos_fallan(monkeypatch):
    router = DataRouter()
    requested = ["AAPL", "MSFT"]

    yahoo = _FakeProvider("Yahoo Fake", prices=pd.DataFrame())
    polygon = _FakeProvider("Polygon Fake", prices=pd.DataFrame())
    fred = _FakeProvider("FRED Fake", prices=pd.DataFrame())

    _install_fakes(router, {"yahoo": yahoo, "polygon": polygon, "fred": fred})

    calls = {"n": 0}
    def _fake_load_cache(tickers):
        calls["n"] += 1
        return _make_multiindex_df(tickers)
    monkeypatch.setattr(router, "_load_cache", _fake_load_cache)

    out = router.get_market_data(requested, period="10y")
    assert not out.empty
    assert calls["n"] == 1
    assert yahoo.calls == 1
    assert polygon.calls == 1
    assert fred.calls == 1