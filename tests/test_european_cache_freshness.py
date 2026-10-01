# -*- coding: utf-8 -*-
"""Fix K: helper comun de cache freshness para providers europeos.

Bug: Euronext y BME usaban `(reference_date - last).days <= 1`. Con
ref=30-sep 23:33 y cache=29-sep, consideran "fresco". No refrescan.
Xetra ya usaba la logica FU-018 (si hoy es sesion del mercado y cerro,
la cache debe contener HOY).

Verificado 2026-09-30: 13 Euronext + 19 BME se quedan en 29-sep.
stock_prices.parquet cae a 89.8% cobertura el 30-sep, market_data
al 100%. Las 12 secciones derivadas del reporte publican datos del 29.

Este test ejercita el PROVIDER real, no solo el helper. Ejerce los
tres contratos:
  - helper european_cache_is_fresh (unitario)
  - EuronextProvider._cache_is_fresh (integracion, con cache en tmp)
  - BMEProvider._cache_is_fresh (idem)
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd


def _make_ref(y, m, d, h=23, mi=33):
    return datetime(y, m, d, h, mi, tzinfo=ZoneInfo("Europe/Madrid"))


# --- Helper ---

def test_european_cache_fresh_con_sesion_cerrada_y_cache_ayer():
    from data.providers._cache_freshness import european_cache_is_fresh
    ref = _make_ref(2026, 9, 30)
    last = pd.Timestamp("2026-09-29")
    assert european_cache_is_fresh(last, "AIR.PA", ref) is False


def test_european_cache_fresh_con_sesion_cerrada_y_cache_hoy():
    from data.providers._cache_freshness import european_cache_is_fresh
    ref = _make_ref(2026, 9, 30)
    last = pd.Timestamp("2026-09-30")
    assert european_cache_is_fresh(last, "AIR.PA", ref) is True


def test_european_cache_fresh_mismo_dia():
    from data.providers._cache_freshness import european_cache_is_fresh
    ref = _make_ref(2026, 9, 29)
    last = pd.Timestamp("2026-09-29")
    assert european_cache_is_fresh(last, "AIR.PA", ref) is True


def test_european_cache_vieja():
    from data.providers._cache_freshness import european_cache_is_fresh
    ref = _make_ref(2026, 9, 30)
    last = pd.Timestamp("2026-09-28")
    assert european_cache_is_fresh(last, "AIR.PA", ref) is False


# --- Euronext provider (integracion) ---

def _write_euronext_cache(tmp_path, ticker, last_date):
    cache_file = tmp_path / f"{ticker.replace('.', '_')}.csv"
    pd.DataFrame({
        "date": [last_date],
        "open": [1.0], "high": [1.0], "low": [1.0],
        "close": [1.0], "volume": [100],
    }).to_csv(cache_file, index=False)
    return cache_file


def test_euronext_provider_cache_ayer_sesion_cerrada_no_fresca(tmp_path, monkeypatch):
    """EuronextProvider._cache_is_fresh con cache=29 y ref=30 -> False."""
    from data.providers import euronext_provider as eup

    monkeypatch.setattr(eup, "EURONEXT_CACHE_DIR", tmp_path)
    _write_euronext_cache(tmp_path, "AIR.PA", "2026-09-29")

    eu = eup.EuronextProvider()
    ref = _make_ref(2026, 9, 30)
    assert eu._cache_is_fresh("AIR.PA", reference_date=ref) is False


def test_euronext_provider_cache_hoy_sesion_cerrada_fresca(tmp_path, monkeypatch):
    """EuronextProvider._cache_is_fresh con cache=30 y ref=30 -> True."""
    from data.providers import euronext_provider as eup

    monkeypatch.setattr(eup, "EURONEXT_CACHE_DIR", tmp_path)
    _write_euronext_cache(tmp_path, "AIR.PA", "2026-09-30")

    eu = eup.EuronextProvider()
    ref = _make_ref(2026, 9, 30)
    assert eu._cache_is_fresh("AIR.PA", reference_date=ref) is True


# --- BME provider (integracion) ---

def _write_bme_cache(tmp_path, ticker, last_date):
    cache_file = tmp_path / f"{ticker.replace('.', '_')}.csv"
    pd.DataFrame({
        "date": [last_date],
        "open": [1.0], "high": [1.0], "low": [1.0],
        "close": [1.0], "volume": [100],
    }).to_csv(cache_file, index=False)
    return cache_file


def test_bme_provider_cache_ayer_sesion_cerrada_no_fresca(tmp_path, monkeypatch):
    """BMEProvider._cache_is_fresh con cache=29 y ref=30 -> False."""
    from data.providers import bme_provider as bmep

    monkeypatch.setattr(bmep, "BME_CACHE_DIR", tmp_path)
    _write_bme_cache(tmp_path, "SAN.MC", "2026-09-29")

    bm = bmep.BMEProvider()
    ref = _make_ref(2026, 9, 30)
    assert bm._cache_is_fresh("SAN.MC", reference_date=ref) is False


def test_bme_provider_cache_hoy_sesion_cerrada_fresca(tmp_path, monkeypatch):
    """BMEProvider._cache_is_fresh con cache=30 y ref=30 -> True."""
    from data.providers import bme_provider as bmep

    monkeypatch.setattr(bmep, "BME_CACHE_DIR", tmp_path)
    _write_bme_cache(tmp_path, "SAN.MC", "2026-09-30")

    bm = bmep.BMEProvider()
    ref = _make_ref(2026, 9, 30)
    assert bm._cache_is_fresh("SAN.MC", reference_date=ref) is True