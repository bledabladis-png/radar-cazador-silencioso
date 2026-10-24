"""Tests pipeline: sector_metrics (fase 6b).

Cubre las 5 sub-funciones del orquestador. Los modulos subyacentes
(indicators.sector_*) se mockean. Se verifica:
  - Contrato: cada sub-funcion devuelve df o None.
  - Persistencia CSV cuando hay df no vacio.
  - Degradacion cuando df_stocks / leader_df son None o vacios.
  - FU-008-b: _compute_concentration NO depende de leader_df.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import sector_metrics as sm


def _setup_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)


def _df_stocks():
    idx = pd.date_range("2026-09-25", periods=1, freq="D")
    cols = pd.MultiIndex.from_product([["Close"], ["AAA"]], names=["field", "ticker"])
    return pd.DataFrame([1.0], index=idx, columns=cols)


def _df_leader():
    return pd.DataFrame([{"sector": "XLK", "ticker": "AAA", "rs": 1.0,
                          "rs_mom": 0.1, "flow_proxy_z": 0.3,
                          "wls": 0.5, "wyckoff_phase": "MARKUP"}])


def _df_fake(extra_cols=None):
    base = {"date": pd.Timestamp("2026-09-25"), "sector": "XLK"}
    if extra_cols:
        base.update(extra_cols)
    return pd.DataFrame([base])


# =============================================================================
# _compute_wyckoff
# =============================================================================

def test_wyckoff_df_stocks_none():
    assert sm._compute_wyckoff(None, pd.DataFrame()) is None


def test_wyckoff_ok(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    with patch("src.pipeline.sector_metrics.compute_sector_wyckoff_distribution",
               return_value=_df_fake()):
        out = sm._compute_wyckoff(_df_stocks(), pd.DataFrame())
    assert out is not None
    p = tmp_path / "outputs" / "history" / "sector_wyckoff_distribution.csv"
    assert p.exists()


def test_wyckoff_error(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    with patch("src.pipeline.sector_metrics.compute_sector_wyckoff_distribution",
               side_effect=RuntimeError("x")):
        out = sm._compute_wyckoff(_df_stocks(), pd.DataFrame())
    assert out is None


# =============================================================================
# _compute_rs_internal
# =============================================================================

def test_rs_internal_df_stocks_none():
    assert sm._compute_rs_internal(None, pd.DataFrame(), pd.DataFrame()) is None


def test_rs_internal_ok(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    with patch("src.pipeline.sector_metrics.compute_rs_internal",
               return_value=_df_fake({"ticker": "AAA"})):
        out = sm._compute_rs_internal(_df_stocks(), pd.DataFrame(), pd.DataFrame())
    assert out is not None
    p = tmp_path / "outputs" / "history" / "rs_internal.csv"
    assert p.exists()


def test_rs_internal_pasa_benchmark_spy(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    captured = {}
    def _fake(df_stocks, holdings_df, df_market, benchmark=None, temporal_meta=None):
        captured["benchmark"] = benchmark
        return _df_fake()
    with patch("src.pipeline.sector_metrics.compute_rs_internal",
               side_effect=_fake):
        sm._compute_rs_internal(_df_stocks(), pd.DataFrame(), pd.DataFrame())
    assert captured["benchmark"] == "SPY"


# =============================================================================
# _compute_concentration
# =============================================================================

def test_concentration_df_stocks_none():
    assert sm._compute_concentration(None, pd.DataFrame(),
                                       pd.DataFrame()) is None


def test_concentration_no_depende_de_leader_df(tmp_path, monkeypatch):
    """FU-008-b: leader_df=None NO bloquea el writer si df_stocks esta ok."""
    _setup_tmp(tmp_path, monkeypatch)
    with patch("src.pipeline.sector_metrics.compute_sector_concentration",
               return_value=_df_fake()):
        out = sm._compute_concentration(_df_stocks(), pd.DataFrame(),
                                          pd.DataFrame(),
                                          reference_date=pd.Timestamp("2026-09-25"))
    assert out is not None
    p = tmp_path / "outputs" / "history" / "sector_concentration.csv"
    assert p.exists()


def test_concentration_dropea_filas_sin_date(tmp_path, monkeypatch):
    """Si compute_sector_concentration devuelve filas con date=NaT, se dropean."""
    _setup_tmp(tmp_path, monkeypatch)
    fake = pd.DataFrame([
        {"date": pd.Timestamp("2026-09-25"), "sector": "XLK"},
        {"date": pd.NaT, "sector": "XLF"},
    ])
    with patch("src.pipeline.sector_metrics.compute_sector_concentration",
               return_value=fake):
        out = sm._compute_concentration(_df_stocks(), pd.DataFrame(),
                                          pd.DataFrame(),
                                          reference_date=pd.Timestamp("2026-09-25"))
    assert out is not None
    # Solo la fila con date valida
    assert len(out) == 1


# =============================================================================
