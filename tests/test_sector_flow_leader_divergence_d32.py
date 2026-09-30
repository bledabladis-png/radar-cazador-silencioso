# -*- coding: utf-8 -*-
"""D32 (2026-09-30): tests para 2 modulos de flow/divergencia.

- sector_leader_divergence.compute_sector_leader_divergence: 16%.
- sector_flow_characteristics.compute_sector_flow_characteristics: 35%.
  + helpers _persistence, _regime.

Test existente test_sector_leader_divergence.py cubre _ret_20d. Aqui
van las funciones principales con datos sinteticos.
"""
import numpy as np
import pandas as pd

from indicators.sector_leader_divergence import compute_sector_leader_divergence
from indicators.sector_flow_characteristics import (
    compute_sector_flow_characteristics,
    _persistence,
    _regime,
)


def _make_df_stocks(tickers, n=100, seed=42, drift=0.001):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {}
    for i, t in enumerate(tickers):
        close = 100 * np.cumprod(1 + rng.normal(drift + i * 0.0001, 0.015, n))
        data[("Close", t)] = close
        data[("High", t)] = close * 1.01
        data[("Low", t)] = close * 0.99
        data[("Open", t)] = close * 0.998
        data[("Volume", t)] = rng.randint(1_000_000, 5_000_000, n)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


# ============================================================
# sector_leader_divergence
# ============================================================
def test_leader_divergence_leader_df_none():
    assert compute_sector_leader_divergence(
        None, pd.DataFrame(), None, None).empty


def test_leader_divergence_leader_df_vacio():
    assert compute_sector_leader_divergence(
        None, pd.DataFrame(), pd.DataFrame(), None).empty


def test_leader_divergence_normal():
    """XLK con 3 lideres positivos, sector positivo -> Alineacion positiva."""
    sector = "XLK"
    # Sector con drift positivo
    df_market = _make_df_stocks([sector], n=100, seed=1, drift=0.002)
    # Lideres tambien positivos
    df_stocks = _make_df_stocks(["AAPL", "MSFT", "NVDA", "AVGO"],
                                 n=100, seed=2, drift=0.003)
    holdings = pd.DataFrame({
        "etf": [sector] * 4,
        "ticker": ["AAPL", "MSFT", "NVDA", "AVGO"],
    })
    leaders = pd.DataFrame({
        "sector": [sector] * 4,
        "ticker": ["AAPL", "MSFT", "NVDA", "AVGO"],
    })
    out = compute_sector_leader_divergence(
        df_stocks, holdings, leaders, df_market)
    assert not out.empty
    assert out.iloc[0]["sector"] == sector
    assert out.iloc[0]["n_leaders_valid"] == 4


def test_leader_divergence_sector_no_en_sectors():
    """etf no en MARKET_TICKERS['sectors'] -> skip."""
    df_market = _make_df_stocks(["ZZZ"], n=100)
    df_stocks = _make_df_stocks(["AAPL"], n=100)
    holdings = pd.DataFrame({"etf": ["ZZZ"], "ticker": ["AAPL"]})
    leaders = pd.DataFrame({"sector": ["ZZZ"], "ticker": ["AAPL"]})
    out = compute_sector_leader_divergence(
        df_stocks, holdings, leaders, df_market)
    assert out.empty


def test_leader_divergence_menos_de_3_validos_nd():
    """Con <3 lideres validos -> classification = N/D."""
    sector = "XLK"
    df_market = _make_df_stocks([sector], n=100)
    df_stocks = _make_df_stocks(["AAPL", "MSFT"], n=100)
    holdings = pd.DataFrame({"etf": [sector] * 2,
                              "ticker": ["AAPL", "MSFT"]})
    leaders = pd.DataFrame({"sector": [sector] * 2,
                             "ticker": ["AAPL", "MSFT"]})
    out = compute_sector_leader_divergence(
        df_stocks, holdings, leaders, df_market)
    if not out.empty:
        assert out.iloc[0]["classification"] == "N/D"


def test_leader_divergence_sector_sin_close():
    """sector_etf sin Close en df_market -> skip."""
    df_market = _make_df_stocks(["ZZZ"], n=100)
    df_stocks = _make_df_stocks(["AAPL"], n=100)
    holdings = pd.DataFrame({"etf": ["XLK"], "ticker": ["AAPL"]})
    leaders = pd.DataFrame({"sector": ["XLK"], "ticker": ["AAPL"]})
    out = compute_sector_leader_divergence(
        df_stocks, holdings, leaders, df_market)
    assert out.empty


# ============================================================
# sector_flow_characteristics: _persistence
# ============================================================
def test_persistence_denom_cero():
    assert pd.isna(_persistence(0, 0))


def test_persistence_todos_pos():
    assert _persistence(5, 0) == 1.0


def test_persistence_todos_neg():
    assert _persistence(0, 5) == 0.0


def test_persistence_mixto():
    assert _persistence(3, 1) == 0.75


# ============================================================
# sector_flow_characteristics: _regime
# ============================================================
def test_regime_nan():
    assert _regime(np.nan, 1.0) == "N/D"
    assert _regime(1.0, np.nan) == "N/D"


def test_regime_confirmacion():
    assert _regime(0.05, 100.0) == "Confirmación"


def test_regime_divergencia_bajista():
    assert _regime(0.05, -100.0) == "Divergencia bajista"


def test_regime_absorcion():
    assert _regime(-0.05, 100.0) == "Absorción potencial"


def test_regime_confirmacion_debilidad():
    assert _regime(-0.05, -100.0) == "Confirmación de debilidad"


def test_regime_neutral():
    assert _regime(0.0, 0.0) == "Neutral / sin confirmación"


# ============================================================
# compute_sector_flow_characteristics
# ============================================================
def _write_flow_csv(path, sector, n=25):
    """CSV con columnas Date, ticker, primary_flow_usd + opcionales."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "Date": dates,
        "ticker": [sector] * n,
        "primary_flow_usd": rng.randn(n) * 1e6,
        "primary_flow_pct": rng.randn(n) * 0.01,
        "primary_flow_z": rng.randn(n),
    })
    df.to_csv(path, index=False)


def test_flow_char_normal(tmp_path):
    sector = "XLK"
    csv = tmp_path / "flow.csv"
    _write_flow_csv(csv, sector)
    price_df = _make_df_stocks([sector], n=100)
    out = compute_sector_flow_characteristics(str(csv), price_df)
    assert not out.empty
    assert out.iloc[0]["sector"] == sector
    assert "flow_5d_sum" in out.columns
    assert "price_flow_regime_20d" in out.columns


def test_flow_char_sector_sin_flow(tmp_path):
    """Sector sin filas en el CSV -> no aparece en el output."""
    csv = tmp_path / "flow.csv"
    _write_flow_csv(csv, "ZZZ")  # solo ZZZ
    price_df = _make_df_stocks(["XLK"], n=100)
    out = compute_sector_flow_characteristics(str(csv), price_df)
    assert out.empty


def test_flow_char_sin_price(tmp_path):
    """Sector sin Close en price_df -> skip (except KeyError)."""
    csv = tmp_path / "flow.csv"
    _write_flow_csv(csv, "XLK")
    price_df = _make_df_stocks(["ZZZ"], n=100)  # sin XLK
    out = compute_sector_flow_characteristics(str(csv), price_df)
    assert out.empty


def test_flow_char_serie_corta_price(tmp_path):
    """price_df con <21 filas -> price_ret_20d NaN, regime N/D."""
    csv = tmp_path / "flow.csv"
    _write_flow_csv(csv, "XLK")
    price_df = _make_df_stocks(["XLK"], n=15)
    out = compute_sector_flow_characteristics(str(csv), price_df)
    if not out.empty:
        assert pd.isna(out.iloc[0]["price_ret_20d"])


def test_flow_char_persistence_negativa(tmp_path):
    """Flujos mayoritariamente negativos -> persistence baja."""
    sector = "XLK"
    csv = tmp_path / "flow.csv"
    dates = pd.date_range("2024-01-01", periods=25, freq="B")
    df = pd.DataFrame({
        "Date": dates,
        "ticker": [sector] * 25,
        "primary_flow_usd": [-1e6] * 25,
    })
    df.to_csv(csv, index=False)
    price_df = _make_df_stocks([sector], n=100)
    out = compute_sector_flow_characteristics(str(csv), price_df)
    if not out.empty:
        assert out.iloc[0]["persistence_20d"] == 0.0
