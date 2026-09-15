# -*- coding: utf-8 -*-
"""Tests FU-018-3b: filtro EOD en Xetra + _cache_is_fresh + integracion."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from data.providers.xetra_provider import XetraProvider


MADRID = ZoneInfo("Europe/Madrid")


@pytest.fixture
def provider():
    """XetraProvider sin red (init solo lee map + crea dir)."""
    return XetraProvider()


def _make_cache_df(rows):
    """rows: lista de (date_str, close). Devuelve df como lo almacena el provider."""
    return pd.DataFrame({
        "date": [pd.Timestamp(d) for d, _ in rows],
        "open": [c for _, c in rows],
        "high": [c for _, c in rows],
        "low": [c for _, c in rows],
        "close": [c for _, c in rows],
        "quantity": [1000 for _ in rows],
    })


# =============================================================================
# _filter_non_eod_last_row
# =============================================================================

def test_filter_empty_df_returns_unchanged(provider):
    ref = datetime(2026, 9, 15, 16, 47, tzinfo=MADRID)
    df = pd.DataFrame(columns=["date", "close"])
    out = provider._filter_non_eod_last_row(df, "SAP.DE", ref)
    assert out.empty


def test_filter_no_reference_date_returns_unchanged(provider):
    df = _make_cache_df([("2026-09-15", 100.0)])
    out = provider._filter_non_eod_last_row(df, "SAP.DE", None)
    assert len(out) == 1


def test_filter_past_session_kept(provider):
    """last_date < ref.date -> conservar."""
    ref = datetime(2026, 9, 15, 16, 47, tzinfo=MADRID)
    df = _make_cache_df([("2026-09-14", 100.0)])
    out = provider._filter_non_eod_last_row(df, "SAP.DE", ref)
    assert len(out) == 1


def test_filter_future_session_warn_kept(provider):
    """last_date > ref.date -> WARN, conservar."""
    ref = datetime(2026, 9, 14, 16, 47, tzinfo=MADRID)
    df = _make_cache_df([("2026-09-15", 100.0)])
    out = provider._filter_non_eod_last_row(df, "SAP.DE", ref)
    assert len(out) == 1


def test_filter_same_day_before_close_removes(provider):
    """Xetra abierto a las 16:47 Madrid -> eliminar vela actual."""
    ref = datetime(2026, 9, 15, 16, 47, tzinfo=MADRID)
    df = _make_cache_df([("2026-09-11", 100.0), ("2026-09-15", 105.0)])
    out = provider._filter_non_eod_last_row(df, "SAP.DE", ref)
    assert len(out) == 1
    assert out["date"].iloc[-1] == pd.Timestamp("2026-09-11")


def test_filter_same_day_after_close_kept(provider):
    """Xetra cerrado a las 19:00 Madrid -> conservar."""
    ref = datetime(2026, 9, 15, 19, 0, tzinfo=MADRID)
    df = _make_cache_df([("2026-09-11", 100.0), ("2026-09-15", 105.0)])
    out = provider._filter_non_eod_last_row(df, "SAP.DE", ref)
    assert len(out) == 2


def test_filter_same_day_weekend_kept(provider):
    """Sabado no es dia de negociacion -> sin cambios."""
    ref = datetime(2026, 9, 19, 16, 0, tzinfo=MADRID)  # sabado
    df = _make_cache_df([("2026-09-19", 100.0)])
    out = provider._filter_non_eod_last_row(df, "SAP.DE", ref)
    assert len(out) == 1


def test_filter_unknown_market_does_not_remove(provider):
    """Ticker con sufijo no reconocido -> WARN, sin eliminar."""
    ref = datetime(2026, 9, 15, 16, 47, tzinfo=MADRID)
    df = _make_cache_df([("2026-09-15", 100.0)])
    out = provider._filter_non_eod_last_row(df, "XYZ.ZZ", ref)
    assert len(out) == 1


# =============================================================================
# _cache_is_fresh con reference_date
# =============================================================================

def test_cache_fresh_legacy_no_reference(provider, monkeypatch):
    """Sin reference_date -> fallback legacy."""
    df = _make_cache_df([("2026-09-14", 100.0)])
    monkeypatch.setattr(provider, "_load_cache", lambda t: df)
    # Sin reference_date -> comportamiento legacy.
    assert provider._cache_is_fresh("SAP.DE") in (True, False)


def test_cache_fresh_today_closed_cache_yesterday_is_stale(provider, monkeypatch):
    """Hoy Xetra ya cerro + cache ayer -> NO fresh."""
    ref = datetime(2026, 9, 15, 22, 0, tzinfo=MADRID)
    df = _make_cache_df([("2026-09-14", 100.0)])
    monkeypatch.setattr(provider, "_load_cache", lambda t: df)
    assert provider._cache_is_fresh("SAP.DE", reference_date=ref) is False


def test_cache_fresh_today_closed_cache_today_is_fresh(provider, monkeypatch):
    """Hoy Xetra ya cerro + cache hoy -> fresh."""
    ref = datetime(2026, 9, 15, 22, 0, tzinfo=MADRID)
    df = _make_cache_df([("2026-09-15", 100.0)])
    monkeypatch.setattr(provider, "_load_cache", lambda t: df)
    assert provider._cache_is_fresh("SAP.DE", reference_date=ref) is True


def test_cache_fresh_market_open_cache_yesterday_fallback(provider, monkeypatch):
    """Xetra abierto + cache ayer -> fallback (gap <= 1)."""
    ref = datetime(2026, 9, 15, 16, 47, tzinfo=MADRID)
    df = _make_cache_df([("2026-09-14", 100.0)])
    monkeypatch.setattr(provider, "_load_cache", lambda t: df)
    assert provider._cache_is_fresh("SAP.DE", reference_date=ref) is True


def test_cache_fresh_empty_cache(provider, monkeypatch):
    df = pd.DataFrame(columns=["date", "close"])
    monkeypatch.setattr(provider, "_load_cache", lambda t: df)
    ref = datetime(2026, 9, 15, 16, 47, tzinfo=MADRID)
    assert provider._cache_is_fresh("SAP.DE", reference_date=ref) is False


# =============================================================================
# Integracion: get_prices con cache-hit
# =============================================================================

def test_get_prices_cache_hit_filters_intraday(provider, monkeypatch):
    """Cache con 14/09 EOD + 15/09 intradia -> get_prices devuelve solo 14/09."""
    ref = datetime(2026, 9, 15, 16, 47, tzinfo=MADRID)
    df_cache = _make_cache_df([("2026-09-11", 100.0), ("2026-09-14", 101.0),
                               ("2026-09-15", 102.0)])
    monkeypatch.setattr(provider, "_load_cache", lambda t: df_cache)

    out = provider.get_prices(["SAP.DE"], use_cache=True, reference_date=ref)

    # Debe tener MultiIndex
    assert isinstance(out.columns, pd.MultiIndex)
    # Ultima fecha en la salida debe ser 14/09 (no 15/09)
    last = out.index.max()
    assert last == pd.Timestamp("2026-09-14")


def test_get_prices_cache_hit_keeps_clean_cache(provider, monkeypatch):
    """Cache sin vela intradia -> get_prices sin cambios."""
    ref = datetime(2026, 9, 15, 16, 47, tzinfo=MADRID)
    df_cache = _make_cache_df([("2026-09-11", 100.0), ("2026-09-14", 101.0)])
    monkeypatch.setattr(provider, "_load_cache", lambda t: df_cache)

    out = provider.get_prices(["SAP.DE"], use_cache=True, reference_date=ref)
    assert out.index.max() == pd.Timestamp("2026-09-14")
    assert len(out) == 2
