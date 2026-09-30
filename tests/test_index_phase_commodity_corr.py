# -*- coding: utf-8 -*-
"""D26 (2026-09-30): tests para 2 modulos indicators con <25% cobertura.

- index_phase.compute_index_phases: 11%. Recibe df_market + usa
  DataRouter como fallback. Se testea con df_market sintetico y, para
  la rama de fallback, mockeando el router (sin red real).
- commodity_market_correlation.compute_commodity_market_correlation:
  10%. Funcion pura. DataFrames sinteticos.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd

from indicators.index_phase import compute_index_phases
from indicators.commodity_market_correlation import compute_commodity_market_correlation
from config.index_tickers import INDEX_CONFIG


def _make_market_df(tickers, n=300, seed=42):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {}
    for i, t in enumerate(tickers):
        drift = 0.0001 * (i + 1)
        close = 100 * np.cumprod(1 + rng.normal(drift, 0.015, n))
        data[("Close", t)] = close
        data[("High", t)] = close * 1.01
        data[("Low", t)] = close * 0.99
        data[("Open", t)] = close * 0.998
        data[("Volume", t)] = rng.randint(1_000_000, 5_000_000, n)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


# ---------- compute_index_phases ----------
def test_index_phases_todos_presentes():
    """Todos los indices en df_market -> phases con todos, sin router."""
    index_tickers = [cfg["index_ticker"] for cfg in INDEX_CONFIG.values()]
    df = _make_market_df(index_tickers, n=300)
    with patch("indicators.index_phase.DataRouter") as mock_router:
        phases, index_data = compute_index_phases(df)
    # No se llama al router si estan todos
    assert not mock_router.return_value.get_market_data.called
    assert index_data is None
    assert len(phases) == len(INDEX_CONFIG)
    for nombre in INDEX_CONFIG:
        assert nombre in phases
        assert isinstance(phases[nombre], str)


def test_index_phases_faltan_tickers_usa_router():
    """Faltan indices -> router.get_market_data se invoca."""
    index_tickers = [cfg["index_ticker"] for cfg in INDEX_CONFIG.values()]
    # Dejar solo el primer ticker en df_market
    df = _make_market_df(index_tickers[:1], n=300)
    fake_data = _make_market_df(index_tickers, n=300)

    with patch("indicators.index_phase.DataRouter") as mock_router_cls:
        instance = mock_router_cls.return_value
        instance.get_market_data.return_value = fake_data
        phases, index_data = compute_index_phases(df)
    assert instance.get_market_data.called
    assert index_data is not None
    # Todos los indices deben tener fase (o ERROR)
    assert len(phases) == len(INDEX_CONFIG)


def test_index_phases_router_falla_devuelve_error():
    """Router lanza excepcion -> phases con ERROR en los faltantes."""
    index_tickers = [cfg["index_ticker"] for cfg in INDEX_CONFIG.values()]
    df = _make_market_df(index_tickers[:1], n=300)

    with patch("indicators.index_phase.DataRouter") as mock_router_cls:
        instance = mock_router_cls.return_value
        instance.get_market_data.side_effect = OSError("sin red")
        phases, index_data = compute_index_phases(df)
    # Todos los que faltaban deben tener 'ERROR'
    error_count = sum(1 for v in phases.values() if v == "ERROR")
    assert error_count > 0


def test_index_phases_vacio_devuelve_todos_error():
    """df_market vacio -> todos van al router -> todos ERROR si falla.

    Verifica explicitamente que phases tiene TODAS las claves de
    INDEX_CONFIG con valor 'ERROR'. El assert `all(...)` sobre dict
    vacio devolveria True: hay que comprobar el tamano tambien.
    """
    empty_df = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples([("Close", "^GSPC")])
    )
    with patch("indicators.index_phase.DataRouter") as mock_router_cls:
        instance = mock_router_cls.return_value
        instance.get_market_data.side_effect = OSError("sin red")
        phases, _ = compute_index_phases(empty_df)
    assert len(phases) == len(INDEX_CONFIG), (
        "phases tiene {} claves, esperado {}".format(
            len(phases), len(INDEX_CONFIG)))
    assert all(v == "ERROR" for v in phases.values())


# ---------- compute_commodity_market_correlation ----------
def test_commodity_corr_normal():
    df = _make_market_df(["XLK", "^GSPC", "^SPGSCI"], n=300)
    out = compute_commodity_market_correlation(df, "XLK")
    assert "commodity_corr" in out
    assert "market_corr" in out
    assert out["commodity_level"] in ("HIGH", "MODERATE", "LOW", "N/A")
    assert out["market_level"] in ("HIGH", "MODERATE", "LOW", "N/A")


def test_commodity_corr_ticker_ausente():
    """Sin la columna del sector -> dict con None."""
    df = _make_market_df(["^GSPC", "^SPGSCI"], n=300)  # sin XLK
    out = compute_commodity_market_correlation(df, "XLK")
    assert out["commodity_corr"] is None
    assert out["market_corr"] is None
    assert out["commodity_level"] == "N/A"
    assert out["market_level"] == "N/A"


def test_commodity_corr_serie_corta():
    """<window observaciones -> None."""
    df = _make_market_df(["XLK", "^GSPC", "^SPGSCI"], n=100)  # < 126
    out = compute_commodity_market_correlation(df, "XLK")
    assert out["commodity_corr"] is None
    assert out["commodity_level"] == "N/A"


def test_commodity_corr_niveles():
    """Verificar que classify_corr produce el nivel correcto segun |corr|."""
    df = _make_market_df(["XLK", "^GSPC", "^SPGSCI"], n=300)
    out = compute_commodity_market_correlation(df, "XLK")
    for key, val, lvl in [
        ("commodity_corr", out["commodity_corr"], out["commodity_level"]),
        ("market_corr", out["market_corr"], out["market_level"]),
    ]:
        if val is None:
            assert lvl == "N/A"
        elif abs(val) > 0.6:
            assert lvl == "HIGH"
        elif abs(val) > 0.3:
            assert lvl == "MODERATE"
        else:
            assert lvl == "LOW"


def test_commodity_corr_ticker_custom():
    """Parametros custom (benchmark, commodity, window) se respetan."""
    df = _make_market_df(["XLK", "^GSPC", "TLT", "GLD"], n=300)
    out = compute_commodity_market_correlation(
        df, "XLK", benchmark="TLT", commodity="GLD", window=60,
    )
    assert out["commodity_corr"] is not None
    assert out["market_corr"] is not None
