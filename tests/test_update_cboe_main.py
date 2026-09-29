"""Tests F3-17-extendido: update_cboe.main() detecta fallo silencioso.

Bug cerrado: main() ignoraba el retorno de fetch_and_write y devolvia
siempre 0. Si el writer fallaba (excepcion capturada dentro del writer
-> {}), el step salia verde. Fix: exit 1 si not ok.

continue-on-error: true en daily_run.yml evita romper el pipeline.
Simetria con update_futures.py (return 1 si esperado pero no recuperado).

Sin red. Provider mockeado. _already_up_to_date forzado a False para
llegar siempre a la rama de fetch.
"""
from __future__ import annotations

from scripts.update_cboe import main


class _FakeProviderOK:
    def __init__(self, result):
        self._result = result

    def fetch_and_write(self, **kwargs):
        return self._result


class _FakeProviderRaise:
    def __init__(self, exc):
        self._exc = exc

    def fetch_and_write(self, **kwargs):
        raise self._exc


def _patch_skip(monkeypatch):
    monkeypatch.setattr(
        "scripts.update_cboe._already_up_to_date",
        lambda *a, **k: False,
    )


def test_fetch_ok_devuelve_0(monkeypatch):
    _patch_skip(monkeypatch)
    monkeypatch.setattr(
        "scripts.update_cboe.CboeIndexProvider",
        lambda: _FakeProviderOK({"parquet": "data/cboe_vix3m.parquet"}),
    )
    assert main() == 0


def test_fetch_vacio_devuelve_1(monkeypatch):
    _patch_skip(monkeypatch)
    monkeypatch.setattr(
        "scripts.update_cboe.CboeIndexProvider",
        lambda: _FakeProviderOK({}),
    )
    assert main() == 1


def test_excepcion_permanente_devuelve_2(monkeypatch):
    _patch_skip(monkeypatch)
    monkeypatch.setattr(
        "scripts.update_cboe.CboeIndexProvider",
        lambda: _FakeProviderRaise(RuntimeError("401 unauthorized")),
    )
    assert main() == 2


def test_excepcion_transitoria_devuelve_0(monkeypatch):
    _patch_skip(monkeypatch)
    monkeypatch.setattr(
        "scripts.update_cboe.CboeIndexProvider",
        lambda: _FakeProviderRaise(TimeoutError("network timeout")),
    )
    assert main() == 0