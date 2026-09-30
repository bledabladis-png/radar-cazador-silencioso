# -*- coding: utf-8 -*-
"""D27 (2026-09-30): tests para indicators/index_leaders.py y
indicators/darkpool_history.py.

- index_leaders: 12% -> compute_wls_for_index (pura) + tests de
  compute_stock_metrics_for_index con df_stocks + df_index_data
  sinteticos (evita el router).
- darkpool_history: 8% -> _backfill_history con finra y yf mockeados.
  Sin red real. Verifica el control de bucle (dedup, tope MAX_PER_RUN,
  iteracion hacia atras).
"""
from unittest.mock import patch

import numpy as np
import pandas as pd

from indicators.index_leaders import (
    compute_wls_for_index,
    compute_stock_metrics_for_index,
)
from indicators.darkpool_history import _backfill_history


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


# ---------- compute_wls_for_index ----------
def _make_metrics_df():
    """Fixture en orden NO ordenado por WLS (evita que un test ciego
    a sort_values pase sin el fix). El orden por calidad esperado es
    A > B > C > D > E, pero las filas van intercaladas.
    """
    return pd.DataFrame({
        "ticker": ["C", "A", "E", "B", "D"],
        "rs": [0.8, 1.2, 0.3, 1.0, 0.5],
        "rs_mom": [0.02, 0.05, -0.01, 0.03, 0.01],
        "flow_proxy_z": [0.0, 1.5, -1.5, 0.5, -0.5],
        "wyckoff_score": [0.5, 0.8, 0.2, 0.6, 0.4],
        "wyckoff_phase": ["MARKUP"] * 5,
        "persistence_10d": [0.5, 0.7, 0.3, 0.6, 0.4],
        "stability": [0.5, 0.9, 0.1, 0.7, 0.3],
    })


def test_wls_vacio_devuelve_vacio():
    out = compute_wls_for_index(pd.DataFrame())
    assert out.empty


def test_wls_columnas_anadidas():
    df = _make_metrics_df()
    out = compute_wls_for_index(df)
    for c in ("rs_z", "flow_proxy_z_norm", "rws_z", "stab_z", "wls"):
        assert c in out.columns


def test_wls_ordenado_descendente():
    df = _make_metrics_df()
    out = compute_wls_for_index(df)
    assert list(out["wls"]) == sorted(out["wls"], reverse=True)


def test_wls_primero_es_el_mejor():
    """El ticker con mejores metricas (A) debe quedar primero."""
    df = _make_metrics_df()
    out = compute_wls_for_index(df)
    assert out.iloc[0]["ticker"] == "A"


def test_wls_zscore_mad_cero():
    """Todos los valores iguales -> MAD=0 -> z=0."""
    df = _make_metrics_df()
    df["rs"] = [1.0] * 5
    out = compute_wls_for_index(df)
    assert (out["rs_z"] == 0.0).all()


def test_wls_pesos_suma_1():
    """Los pesos del WLS suman 1.0 (0.35 + 0.25 + 0.25 + 0.10)."""
    df = _make_metrics_df()
    out = compute_wls_for_index(df)
    # Verifico contra el calculo manual
    for _, row in out.iterrows():
        manual = (0.35 * row["rs_z"] + 0.25 * row["flow_proxy_z_norm"]
                  + 0.25 * row["rws_z"] + 0.10 * row["stab_z"])
        manual *= 1 + 0.05 * min(row["persistence_10d"], 1.0)
        assert abs(row["wls"] - manual) < 1e-9


# ---------- compute_stock_metrics_for_index ----------
def test_compute_stock_metrics_con_df_index_data():
    """df_index_data provisto con el index_ticker real -> no llama router."""
    from config.index_tickers import INDEX_CONFIG
    etf_ticker = INDEX_CONFIG["Nasdaq-100"]["index_ticker"]
    df_stocks = _make_market_df(["AAPL", "MSFT"], n=300)
    df_index = _make_market_df([etf_ticker], n=300)
    with patch("indicators.index_leaders.DataRouter") as mock_router:
        out = compute_stock_metrics_for_index(
            df_stocks, "Nasdaq-100", ["AAPL", "MSFT"], df_index_data=df_index)
    assert not mock_router.return_value.get_market_data.called
    assert isinstance(out, pd.DataFrame)


def test_compute_stock_metrics_sin_df_stocks():
    """df_stocks None -> DataFrame vacio (loop salta todos)."""
    from config.index_tickers import INDEX_CONFIG
    etf_ticker = INDEX_CONFIG["Nasdaq-100"]["index_ticker"]
    df_index = _make_market_df([etf_ticker], n=300)
    out = compute_stock_metrics_for_index(
        None, "Nasdaq-100", ["AAPL"], df_index_data=df_index)
    assert out.empty


def test_compute_stock_metrics_ticker_no_en_df():
    """Ticker no presente -> skip, DataFrame vacio."""
    from config.index_tickers import INDEX_CONFIG
    etf_ticker = INDEX_CONFIG["Nasdaq-100"]["index_ticker"]
    df_stocks = _make_market_df(["AAPL"], n=300)
    df_index = _make_market_df([etf_ticker], n=300)
    out = compute_stock_metrics_for_index(
        df_stocks, "Nasdaq-100", ["NOEXISTE"], df_index_data=df_index)
    assert out.empty


# ---------- darkpool_history._backfill_history ----------
class _FakeFinra:
    def __init__(self, latest_week, ats_by_week):
        self._latest = latest_week
        self._ats = ats_by_week

    def get_latest_week(self):
        return self._latest

    def get_all_tiers(self, week_str):
        return self._ats.get(week_str, pd.DataFrame())


def test_backfill_hist_suficiente_no_descarga():
    """104 semanas ya presentes -> return hist sin tocar."""
    hist = pd.DataFrame({"week": pd.date_range("2024-01-01", periods=104, freq="W"),
                         "ratio": np.random.rand(104)})
    fake = _FakeFinra("2026-09-26", {})
    out = _backfill_history(hist, fake)
    assert len(out) == 104


def test_backfill_no_latest_week():
    """finra no devuelve semana -> return hist sin cambios."""
    hist = pd.DataFrame({"week": pd.date_range("2024-01-01", periods=50, freq="W"),
                         "ratio": np.random.rand(50)})
    fake = _FakeFinra(None, {})
    out = _backfill_history(hist, fake)
    assert len(out) == 50


def test_backfill_sin_volumen_yf():
    """finra devuelve datos pero yf no da volumen -> no añade filas."""
    hist = pd.DataFrame({"week": pd.date_range("2024-01-01", periods=50, freq="W"),
                         "ratio": np.random.rand(50)})
    ats = pd.DataFrame({"issueSymbolIdentifier": ["AAPL"],
                        "totalWeeklyShareQuantity": [1000]})
    fake = _FakeFinra("2026-09-26", {"2026-09-19": ats})
    with patch("indicators.darkpool_history._get_all_tickers", return_value=["AAPL"]):
        with patch("indicators.darkpool_history.yf") as mock_yf:
            mock_yf.download.return_value = pd.DataFrame()  # vacio
            out = _backfill_history(hist, fake)
    assert len(out) == 50


def test_backfill_anade_una_fila():
    """finra + yf OK -> añade 1 fila (MAX_PER_RUN=1)."""
    hist = pd.DataFrame({"week": pd.date_range("2024-01-01", periods=50, freq="W"),
                         "ratio": np.random.rand(50)})
    ats = pd.DataFrame({"issueSymbolIdentifier": ["AAPL"],
                        "totalWeeklyShareQuantity": [5000]})
    fake = _FakeFinra("2026-09-26", {"2026-09-19": ats})
    yf_data = pd.DataFrame({"Volume": [100000]}, index=pd.date_range("2026-09-19", periods=5))
    with patch("indicators.darkpool_history._get_all_tickers", return_value=["AAPL"]):
        with patch("indicators.darkpool_history.yf") as mock_yf:
            mock_yf.download.return_value = yf_data
            out = _backfill_history(hist, fake)
    # MAX_PER_RUN = 1
    assert len(out) == 51
    assert out["week"].is_monotonic_increasing
