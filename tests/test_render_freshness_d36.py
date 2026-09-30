# -*- coding: utf-8 -*-
"""D36 (2026-09-30): tests de src/report/freshness.render_data_freshness.

El modulo tiene 65% de cobertura. Las ramas sin cubrir son las de
cada fuente cuando HAY datos: CBOE, FINRA, FRED y Yahoo Finance.
Aqui van con monkeypatch.chdir a tmp_path para que las rutas
relativas apunten a directorios limpios.
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.report.freshness import render_data_freshness


def _joined(out):
    return "".join(out)


def test_sin_ningun_dato(tmp_path, monkeypatch):
    """Sin pcr_data, sin darkpool, sin liquidity_state, sin parquet."""
    monkeypatch.chdir(tmp_path)
    out = render_data_freshness(None, None, None)
    joined = _joined(out)
    assert "### Data Freshness" in joined
    assert "| CBOE (Opciones) | N/D |" in joined
    assert "| FRED (Macro) | N/D |" in joined
    assert "| Yahoo Finance (Precios) | N/D |" in joined


def test_cboe_con_fecha(tmp_path, monkeypatch):
    """pcr_data con last_date reciente -> CBOE CURRENT."""
    monkeypatch.chdir(tmp_path)
    ref = datetime(2026, 9, 30, 12, 0)
    pcr = {"last_date": "2026-09-29"}
    out = render_data_freshness(pcr, None, None, reference_date=ref)
    joined = _joined(out)
    assert "CBOE (Opciones) | 2026-09-29" in joined
    assert "| 1 dias |" in joined
    assert "CURRENT" in joined


def test_cboe_last_date_invalido(tmp_path, monkeypatch):
    """last_date no parseable -> N/D (rama except)."""
    monkeypatch.chdir(tmp_path)
    pcr = {"last_date": "no-es-fecha"}
    out = render_data_freshness(pcr, None, None)
    joined = _joined(out)
    assert "N/D" in joined
    # La primera columna sigue siendo CBOE
    cboe_lines = [l for l in joined.split(chr(10)) if "CBOE" in l]
    assert any("N/D" in l for l in cboe_lines)


def test_finra_con_fecha(tmp_path, monkeypatch):
    """darkpool_data con week -> FINRA CURRENT/RECENT."""
    monkeypatch.chdir(tmp_path)
    ref = datetime(2026, 9, 30, 12, 0)
    dp = {"week": "2026-09-25"}
    out = render_data_freshness(None, dp, None, reference_date=ref)
    joined = _joined(out)
    assert "FINRA (Dark Pools) | 2026-09-25" in joined


def test_finra_week_invalido(tmp_path, monkeypatch):
    """week no parseable -> N/D."""
    monkeypatch.chdir(tmp_path)
    dp = {"week": "no-fecha"}
    out = render_data_freshness(None, dp, None)
    joined = _joined(out)
    finra_lines = [l for l in joined.split(chr(10)) if "FINRA" in l]
    assert any("N/D" in l for l in finra_lines)


def test_finra_week_na(tmp_path, monkeypatch):
    """week == N/A -> N/D."""
    monkeypatch.chdir(tmp_path)
    dp = {"week": "N/A"}
    out = render_data_freshness(None, dp, None)
    joined = _joined(out)
    finra_lines = [l for l in joined.split(chr(10)) if "FINRA" in l]
    assert any("N/D" in l for l in finra_lines)


def test_fred_con_liquidity_state(tmp_path, monkeypatch):
    """liquidity_state.json con date -> FRED CURRENT."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "state").mkdir(parents=True)
    state = {"date": "2026-09-29"}
    (tmp_path / "outputs" / "state" / "liquidity_state.json").write_text(
        json.dumps(state), encoding="utf-8")
    ref = datetime(2026, 9, 30, 12, 0)
    out = render_data_freshness(None, None, None, reference_date=ref)
    joined = _joined(out)
    # FRED debe tener la fecha (via walk-back al ultimo dia bursatil)
    fred_lines = [l for l in joined.split(chr(10)) if "FRED" in l]
    assert any("2026-09-29" in l or "2026-09-2" in l for l in fred_lines)


def test_fred_state_sin_date(tmp_path, monkeypatch):
    """liquidity_state.json sin date o con N/A -> N/D."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "state").mkdir(parents=True)
    (tmp_path / "outputs" / "state" / "liquidity_state.json").write_text(
        json.dumps({"date": "N/A"}), encoding="utf-8")
    out = render_data_freshness(None, None, None)
    joined = _joined(out)
    fred_lines = [l for l in joined.split(chr(10)) if "FRED" in l]
    assert any("N/D" in l for l in fred_lines)


def test_fred_state_json_invalido(tmp_path, monkeypatch):
    """liquidity_state.json no-JSON -> N/D (rama except)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "state").mkdir(parents=True)
    (tmp_path / "outputs" / "state" / "liquidity_state.json").write_text(
        "no-json", encoding="utf-8")
    out = render_data_freshness(None, None, None)
    joined = _joined(out)
    assert "FRED (Macro) | N/D" in joined


def test_yahoo_con_parquet(tmp_path, monkeypatch):
    """market_data.parquet presente -> Yahoo con fecha."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    idx = pd.date_range("2026-09-01", periods=10, freq="B")
    df = pd.DataFrame({("Close", "SPY"): np.linspace(100, 105, 10)}, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.to_parquet(tmp_path / "data" / "market_data.parquet")
    ref = datetime(2026, 9, 30, 12, 0)
    out = render_data_freshness(None, None, None, reference_date=ref)
    joined = _joined(out)
    yahoo_lines = [l for l in joined.split(chr(10)) if "Yahoo Finance" in l]
    # Debe tener fecha real, no N/D
    assert any("2026-09" in l for l in yahoo_lines)


def test_yahoo_parquet_vacio(tmp_path, monkeypatch):
    """market_data.parquet vacio -> N/D."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    df = pd.DataFrame()
    df.to_parquet(tmp_path / "data" / "market_data.parquet")
    out = render_data_freshness(None, None, None)
    joined = _joined(out)
    yahoo_lines = [l for l in joined.split(chr(10)) if "Yahoo Finance" in l]
    assert any("N/D" in l for l in yahoo_lines)


def test_reference_date_none_fallback(tmp_path, monkeypatch):
    """Sin reference_date -> fallback now()."""
    monkeypatch.chdir(tmp_path)
    pcr = {"last_date": "2026-09-29"}
    out = render_data_freshness(pcr, None, None)
    joined = _joined(out)
    assert "CBOE" in joined
