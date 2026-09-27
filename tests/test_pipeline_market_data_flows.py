"""Tests pipeline: market_data (fase 9b) + flows_secondary (fase 5).

Cobertura de orquestadores. Los modulos subyacentes (indicators.options,
indicators.darkpool, etc.) ya estan cubiertos por sus propios tests.
Aqui se verifica contrato de retorno, manejo de fallos, propagacion de
argumentos y sintesis de flujo.
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.market_data import (
    compute_market_data, _compute_pcr, _compute_darkpool,
    _compute_vol_structure, _compute_data_quality,
)
from src.pipeline.flows_secondary import compute_flows_secondary


# =============================================================================
# market_data.py
# =============================================================================

def test_compute_market_data_devuelve_4_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    with patch("indicators.options.compute_pcr_signals", return_value=None), \
         patch("indicators.darkpool.compute_darkpool_signals", return_value=None), \
         patch("indicators.volatility_structure.compute_volatility_structure",
               return_value=pd.DataFrame()), \
         patch("indicators.data_quality.compute_data_quality",
               return_value=pd.DataFrame()):
        out = compute_market_data(pd.DataFrame())
    assert set(out.keys()) == {
        "pcr_data", "darkpool_data", "vol_structure_df", "data_quality_df"}


def test_compute_pcr_devuelve_none_si_modulo_lanza():
    with patch("indicators.options.compute_pcr_signals",
               side_effect=RuntimeError("x")):
        out = _compute_pcr()
    assert out is None


def test_compute_darkpool_propaga_df_market_y_df_stocks():
    fake = {"media_dark_pool": 22.0, "n_tickers_ats": 536, "n_tickers_total": 536}
    with patch("indicators.darkpool.compute_darkpool_signals",
               return_value=fake) as m:
        out = _compute_darkpool(df_market="M", df_stocks="S")
    assert out == fake
    _, kwargs = m.call_args
    assert kwargs["df_market"] == "M"
    assert kwargs["df_stocks"] == "S"


def test_compute_vol_structure_devuelve_none_si_df_vacio(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    with patch("indicators.volatility_structure.compute_volatility_structure",
               return_value=pd.DataFrame()):
        out = _compute_vol_structure(pd.DataFrame(), pcr_data=None)
    assert out is None


def test_compute_data_quality_devuelve_none_si_df_vacio(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    with patch("indicators.data_quality.compute_data_quality",
               return_value=pd.DataFrame()):
        out = _compute_data_quality()
    assert out is None


# =============================================================================
# flows_secondary.py
# =============================================================================

def _setup_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)


def test_flows_secondary_devuelve_4_keys(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    out = compute_flows_secondary(
        sector_flow_rank=[("XLK", 0.5)],
        etf_primary_flow_data=None,
        cftc_position_flow_data=None,
        blackrock_dax_flow=None,
        blackrock_isf_flow=None,
        amundi_lyxi_flow=None,
    )
    assert set(out.keys()) == {
        "flow_synthesis", "nport_position_change_data",
        "qqq_performance_data", "qqq_nport_flow_data"}


def test_flows_secondary_confidence_alta_cuando_3_signos_coinciden(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    out = compute_flows_secondary(
        sector_flow_rank=[("XLK", 0.5), ("XLF", 0.4)],
        etf_primary_flow_data=pd.DataFrame({"primary_flow_z": [0.5, 0.6]}),
        cftc_position_flow_data=pd.DataFrame({"flow_z": [0.3, 0.4]}),
        blackrock_dax_flow=None,
        blackrock_isf_flow=None,
        amundi_lyxi_flow=None,
    )
    assert out["flow_synthesis"]["confidence"] == "ALTA"


def test_flows_secondary_confidence_media_cuando_2_coinciden(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    out = compute_flows_secondary(
        sector_flow_rank=[("XLK", 0.5)],
        etf_primary_flow_data=pd.DataFrame({"primary_flow_z": [0.5]}),
        cftc_position_flow_data=None,
        blackrock_dax_flow=None,
        blackrock_isf_flow=None,
        amundi_lyxi_flow=None,
    )
    assert out["flow_synthesis"]["confidence"] == "MEDIA"


def test_flows_secondary_confidence_baja_sin_coincidencias(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    out = compute_flows_secondary(
        sector_flow_rank=[("XLK", 0.05)],
        etf_primary_flow_data=None,
        cftc_position_flow_data=None,
        blackrock_dax_flow=None,
        blackrock_isf_flow=None,
        amundi_lyxi_flow=None,
    )
    assert out["flow_synthesis"]["confidence"] == "BAJA"


def test_flows_secondary_cftc_sign_promedio(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    out = compute_flows_secondary(
        sector_flow_rank=None,
        etf_primary_flow_data=None,
        cftc_position_flow_data=pd.DataFrame({"flow_z": [0.2, 0.4, 0.6]}),
        blackrock_dax_flow=None,
        blackrock_isf_flow=None,
        amundi_lyxi_flow=None,
    )
    assert out["flow_synthesis"]["cftc_flow_sign"] == pytest.approx(0.4)


def test_flows_secondary_european_sign_promedio(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    out = compute_flows_secondary(
        sector_flow_rank=None,
        etf_primary_flow_data=None,
        cftc_position_flow_data=None,
        blackrock_dax_flow=pd.DataFrame({"flow_zscore": [0.1, 0.3]}),
        blackrock_isf_flow=pd.DataFrame({"flow_zscore": [-0.2, -0.4]}),
        amundi_lyxi_flow=pd.DataFrame({"flow_zscore": [0.2, 0.5]}),
    )
    # ultimos: 0.3, -0.4, 0.5 -> mean = 0.4/3 ≈ 0.1333
    assert out["flow_synthesis"]["european_flow_sign"] == pytest.approx(0.4 / 3, abs=1e-6)


def test_flows_secondary_qqq_performance_omitido_si_antiguo(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    perf = tmp_path / "outputs" / "history" / "qqq_returns_yahoo.csv"
    perf.write_text("dummy", encoding="utf-8")
    old_ts = (datetime.now() - timedelta(days=30)).timestamp()
    os.utime(perf, (old_ts, old_ts))

    out = compute_flows_secondary(
        sector_flow_rank=None, etf_primary_flow_data=None,
        cftc_position_flow_data=None, blackrock_dax_flow=None,
        blackrock_isf_flow=None, amundi_lyxi_flow=None,
    )
    assert out["qqq_performance_data"] is None


def test_flows_secondary_qqq_performance_cargado_si_reciente(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    perf = tmp_path / "outputs" / "history" / "qqq_returns_yahoo.csv"
    perf.write_text("dummy", encoding="utf-8")
    fresh_ts = datetime.now().timestamp()
    os.utime(perf, (fresh_ts, fresh_ts))

    out = compute_flows_secondary(
        sector_flow_rank=None, etf_primary_flow_data=None,
        cftc_position_flow_data=None, blackrock_dax_flow=None,
        blackrock_isf_flow=None, amundi_lyxi_flow=None,
    )
    assert out["qqq_performance_data"] is not None