"""Tests pipeline: sectors_base (fase 3) + leaders (fase 6a).

Cobertura de orquestadores con fuerte dependencia externa. Los modulos
subyacentes ya estan cubiertos por sus propios tests. Aqui se verifica
contrato de retorno, degradacion individual de cada sub-bloque, y la
logica de HOLIDAY_MODE de leaders.

Nota: leaders.compute_leaders devuelve 6 keys (incluye
df_stocks_effective_meta), pero el docstring solo menciona 5. Se
verifica el contrato REAL (6 keys) y se documenta la discrepancia.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.sectors_base import compute_sectors_base
from src.pipeline.leaders import compute_leaders


def _df_market_min():
    """DataFrame con MultiIndex y 1 fila. Necesario porque compute_sectors_base
    usa df_market.index[-1] como fallback de observacion."""
    idx = pd.date_range("2026-09-25", periods=1, freq="D")
    cols = pd.MultiIndex.from_tuples([("Close", "XLK")], names=["field", "ticker"])
    return pd.DataFrame([1.0], index=idx, columns=cols)


# =============================================================================
# sectors_base.py
# =============================================================================

def _empty_breadth_series():
    idx = pd.date_range("2026-09-20", periods=3, freq="D")
    return pd.Series([0.5, 0.5, 0.5], index=idx)


def test_sectors_base_contract_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    with patch("src.pipeline.sectors_base.compute_sector_scores",
               return_value={"ranking": [("XLK", "Technology", 0.5, "ACCUMULATION")],
                             "regime": "NARROW RALLY"}), \
         patch("src.pipeline.sectors_base.compute_price_flow_rankings",
               return_value=([], [], [], [])), \
         patch("src.pipeline.sectors_base.compute_breadth",
               return_value=(_empty_breadth_series(),) * 5), \
         patch("indicators.sector_rank_history.update_rank_history",
               return_value=(pd.DataFrame(), pd.DataFrame())), \
         patch("indicators.sector_dispersion.compute_sector_dispersion",
               return_value=pd.DataFrame()), \
         patch("indicators.sector_correlation.compute_sector_correlation",
               return_value=pd.DataFrame()), \
         patch("indicators.cross_asset_context.compute_cross_asset_context",
               return_value=(pd.DataFrame(), pd.DataFrame())):
        out = compute_sectors_base(_df_market_min())
    expected_keys = {
        "sector_results", "sector_rank_deltas_df",
        "sector_price_rank", "sector_flow_rank",
        "otros_price_rank", "otros_flow_rank",
        "sector_dispersion_df",
        "sector_corr_summary_df",
        "cross_asset_summary_df", "breadth_values",
    }
    assert set(out.keys()) == expected_keys


def test_sectors_base_breadth_values_derivados(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    idx = pd.date_range("2026-09-20", periods=2, freq="D")
    s = pd.Series([0.5, 0.5], index=idx)
    with patch("src.pipeline.sectors_base.compute_sector_scores",
               return_value={"ranking": [], "regime": "MIXED"}), \
         patch("src.pipeline.sectors_base.compute_price_flow_rankings",
               return_value=([], [], [], [])), \
         patch("src.pipeline.sectors_base.compute_breadth",
               return_value=(s, s, s, s, s)), \
         patch("indicators.sector_rank_history.update_rank_history",
               return_value=(pd.DataFrame(), pd.DataFrame())), \
         patch("indicators.sector_dispersion.compute_sector_dispersion",
               return_value=pd.DataFrame()), \
         patch("indicators.sector_correlation.compute_sector_correlation",
               return_value=pd.DataFrame()), \
         patch("indicators.cross_asset_context.compute_cross_asset_context",
               return_value=(pd.DataFrame(), pd.DataFrame())):
        out = compute_sectors_base(_df_market_min())
    bv = out["breadth_values"]
    # 0.5 * 11 = 5.5 -> round = 6
    assert bv["EMA20 count"] == 6
    assert bv["EMA50 count"] == 6
    assert bv["EMA200 count"] == 6
    assert bv["% sobre EMA20"] == pytest.approx(0.5)


def test_sectors_base_degrada_sub_bloque_dispersion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    idx = pd.date_range("2026-09-20", periods=2, freq="D")
    s = pd.Series([0.5, 0.5], index=idx)
    with patch("src.pipeline.sectors_base.compute_sector_scores",
               return_value={"ranking": [], "regime": "MIXED"}), \
         patch("src.pipeline.sectors_base.compute_price_flow_rankings",
               return_value=([], [], [], [])), \
         patch("src.pipeline.sectors_base.compute_breadth",
               return_value=(s, s, s, s, s)), \
         patch("indicators.sector_rank_history.update_rank_history",
               return_value=(pd.DataFrame(), pd.DataFrame())), \
         patch("indicators.sector_dispersion.compute_sector_dispersion",
               side_effect=RuntimeError("x")), \
         patch("indicators.sector_correlation.compute_sector_correlation",
               return_value=pd.DataFrame()), \
         patch("indicators.cross_asset_context.compute_cross_asset_context",
               return_value=(pd.DataFrame(), pd.DataFrame())):
        out = compute_sectors_base(_df_market_min())
    assert out["sector_dispersion_df"] is None


# =============================================================================
# leaders.py
# =============================================================================

def test_leaders_contract_6_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("src.pipeline.leaders.download_stock_prices", return_value=None):
        out = compute_leaders(pd.DataFrame(), sector_results={"ranking": []})
    # El contrato real tiene 6 keys (docstring menciona 5)
    assert set(out.keys()) == {
        "df_stocks", "df_stocks_effective_meta", "holdings_df",
        "leader_lines", "leader_df", "full_metrics_df"}


def test_leaders_sin_df_stocks_devuelve_todo_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("src.pipeline.leaders.download_stock_prices", return_value=None):
        out = compute_leaders(pd.DataFrame(), sector_results={"ranking": []})
    assert out["df_stocks"] is None
    assert out["leader_lines"] is None
    assert out["leader_df"] is None
    assert out["full_metrics_df"] is None


def test_leaders_holiday_mode_cuando_cobertura_baja(tmp_path, monkeypatch):
    """df_stocks con < 50% Close validos en la ultima fila -> HOLIDAY_MODE."""
    monkeypatch.chdir(tmp_path)
    # leaders lee data/etf_holdings.csv en ruta relativa
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "etf_holdings.csv").write_text(
        "etf,ticker,weight\nXLK,AAA,10.0\n", encoding="utf-8")
    idx = pd.date_range("2026-09-20", periods=5, freq="D")
    cols = pd.MultiIndex.from_tuples(
        [("Close", "AAA"), ("Close", "BBB"), ("Close", "CCC"), ("Close", "DDD")],
        names=["field", "ticker"],
    )
    df = pd.DataFrame(np.ones((5, 4)), index=idx, columns=cols)
    df.iloc[-1, :] = np.nan  # 0/4 validos -> cobertura 0.0

    fake_eff = {
        "status": "OK",
        "date": idx[-1],
        "requested_date": idx[-1],
        "lag_days": 0,
        "coverage": 1.0,
        "n_observed": 4,
        "n_eligible": 4,
    }
    with patch("src.pipeline.leaders.download_stock_prices", return_value=df), \
         patch("src.pipeline.leaders.resolve_effective_date", return_value=fake_eff):
        out = compute_leaders(pd.DataFrame(), sector_results={"ranking": []})
    assert out["df_stocks"] is None
    assert out["leader_lines"] is None
    assert out["leader_df"] is None


def test_leaders_insufficient_coverage_omite_df(tmp_path, monkeypatch):
    """resolve_effective_date con status != OK -> df_stocks = None."""
    monkeypatch.chdir(tmp_path)
    idx = pd.date_range("2026-09-20", periods=3, freq="D")
    cols = pd.MultiIndex.from_tuples([("Close", "AAA")], names=["field", "ticker"])
    df = pd.DataFrame([1.0, 1.0, 1.0], index=idx, columns=cols)

    fake_eff = {
        "status": "INSUFFICIENT_COVERAGE",
        "date": None,
        "requested_date": idx[-1],
        "lag_days": 0,
        "coverage": 0.5,
        "n_observed": 1,
        "n_eligible": 2,
    }
    with patch("src.pipeline.leaders.download_stock_prices", return_value=df), \
         patch("src.pipeline.leaders.resolve_effective_date", return_value=fake_eff):
        out = compute_leaders(pd.DataFrame(), sector_results={"ranking": []})
    assert out["df_stocks"] is None
    assert out["leader_lines"] is None