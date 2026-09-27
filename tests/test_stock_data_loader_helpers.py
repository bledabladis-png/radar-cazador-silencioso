"""Tests de src/stock_data_loader.py (F4-12).

Cubre las funciones no testeadas previamente (helpers + orquestador).
Los helpers especificos ya tienen tests propios:
  - _fill_holes_respecting_sessions: test_b1_session_integrity,
    test_fill_holes_us_holidays, test_fu001_sin_ffill_global.
  - _filter_failed_from_batch: test_fu015_dedup.
  - _filter_non_eod_batch: test_fu018_yahoo_eod.
  - _classify_ticker: test_freshness.
  - _apply_lse_close_override: test_stock_data_loader_lse_override.

Aqui se cubren los helpers no testeados y las ramas principales de
download_stock_prices.
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.stock_data_loader as sdl


# =============================================================================
# Helpers
# =============================================================================

def _multiindex_df(tickers, n=5):
    idx = pd.date_range("2026-09-20", periods=n, freq="D")
    fields = ["Open", "High", "Low", "Close", "Volume"]
    cols = pd.MultiIndex.from_product([fields, tickers], names=["field", "ticker"])
    return pd.DataFrame(np.random.rand(n, len(cols)), index=idx, columns=cols)


def test_get_usa_tickers_sin_fichero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = sdl.get_usa_tickers()
    assert out == []


def test_get_usa_tickers_ordena_por_weight(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "etf_holdings.csv").write_text(
        "etf,ticker,weight\nXLK,A,0.5\nXLK,B,0.9\nXLK,C,0.7\n",
        encoding="utf-8",
    )
    out = sdl.get_usa_tickers()
    assert out == ["B", "C", "A"]


def test_get_stock_list_une_sectores_e_indices(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "etf_holdings.csv").write_text(
        "etf,ticker,weight\nXLK,A,0.9\nXLK,B,0.5\n", encoding="utf-8",
    )
    (tmp_path / "data" / "index_holdings.csv").write_text(
        "etf,ticker,weight\n^GSPC,C,0.8\n^GSPC,D,0.6\n", encoding="utf-8",
    )
    out = sdl.get_stock_list()
    assert "A" in out
    assert "B" in out
    assert "C" in out
    assert "D" in out
    # Sin duplicados
    assert len(out) == len(set(out))


def test_get_stock_list_dedup_preserva_orden(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "etf_holdings.csv").write_text(
        "etf,ticker,weight\nXLK,A,0.9\nXLK,B,0.8\n", encoding="utf-8",
    )
    (tmp_path / "data" / "index_holdings.csv").write_text(
        "etf,ticker,weight\n^GSPC,A,0.9\n^GSPC,C,0.7\n", encoding="utf-8",
    )
    out = sdl.get_stock_list()
    # A viene primero por etf_holdings, C solo en el segundo
    assert out == ["A", "B", "C"]


def test_get_yf_session_devuelve_none_si_no_hay_curl_cffi():
    """Si curl_cffi no esta instalado, retorna None."""
    with patch.dict("sys.modules", {"curl_cffi": None}):
        out = sdl._get_yf_session()
    assert out is None


def test_log_yahoo_raw_diagnostics_df_vacio(capsys):
    sdl._log_yahoo_raw_diagnostics(pd.DataFrame(), None, "batch_x")
    captured = capsys.readouterr()
    assert "batch=batch_x empty=True" in captured.out


def test_log_yahoo_raw_diagnostics_cuenta_nan(capsys):
    idx = pd.date_range("2026-09-25", periods=3, freq="D")
    cols = pd.MultiIndex.from_product(
        [["Close", "Open"], ["AAA", "BBB", "CCC"]],
        names=["field", "ticker"],
    )
    df = pd.DataFrame(np.ones((3, 6)), index=idx, columns=cols)
    # Ultima fila: Close de BBB a NaN -> 1 NaN de 3 Close
    df.loc[idx[-1], ("Close", "BBB")] = np.nan

    ref = datetime.now(ZoneInfo("Europe/Madrid"))
    with patch.object(sdl, "last_expected_market_date",
                      return_value=idx[-1].date()):
        sdl._log_yahoo_raw_diagnostics(df, ref, "batch_1")
    captured = capsys.readouterr()
    assert "batch=batch_1" in captured.out
    assert "close_nan=1/3" in captured.out
    assert "BBB" in captured.out


def test_write_lse_provenance_safe_ok(tmp_path, monkeypatch, capsys):
    captured = {}
    def _fake_write(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
    with patch.object(sdl, "write_lse_provenance", side_effect=_fake_write):
        sdl._write_lse_provenance_safe(
            lse_session="2026-09-25", target_session="2026-09-25",
            run_id="run1",
            stats={"tickers_from_scraper": ["A.L"], "tickers_from_yahoo": [],
                   "scraper_available": True, "tickers_missing": [],
                   "provenance_status": "OK"},
            source_commit="abc123", reason=None,
        )
    assert captured["target_session"] == "2026-09-25"
    assert captured["source_commit"] == "abc123"
    assert captured["status"] == "OK"


def test_write_lse_provenance_safe_value_error_no_aborta(capsys):
    """Si write_lse_provenance lanza ValueError, se loguea y NO se propaga."""
    with patch.object(sdl, "write_lse_provenance",
                      side_effect=ValueError("source_commit obligatorio")):
        sdl._write_lse_provenance_safe(
            lse_session="2026-09-25", target_session="2026-09-25",
            run_id="run1",
            stats={"tickers_from_scraper": ["A.L"], "tickers_from_yahoo": [],
                   "scraper_available": True, "tickers_missing": [],
                   "provenance_status": "OK"},
            source_commit=None, reason=None,
        )
    captured = capsys.readouterr()
    assert "provenance no escrita" in captured.out


# =============================================================================
# download_stock_prices: ramas principales
# =============================================================================

def test_download_reference_date_naive_lanza(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    naive = datetime(2026, 9, 25, 12, 0, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        sdl.download_stock_prices(reference_date=naive)


def test_download_sin_cache_sin_tickers_devuelve_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    ref = datetime.now(ZoneInfo("Europe/Madrid"))
    with patch.object(sdl, "get_stock_list", return_value=[]):
        out = sdl.download_stock_prices(reference_date=ref)
    assert out is None


def test_download_cache_parquet_fresco_devuelve_df(tmp_path, monkeypatch):
    """Cache valido (< CACHE_HOURS) que cubre la sesion esperada -> devuelve df."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    idx = pd.date_range("2026-09-20", periods=5, freq="D")
    cols = pd.MultiIndex.from_product([["Close"], ["AAA"]], names=["field", "ticker"])
    df = pd.DataFrame(np.ones((5, 1)), index=idx, columns=cols)
    pq_path = tmp_path / "data" / "stock_prices.parquet"
    df.to_parquet(pq_path)

    ref = datetime.now(ZoneInfo("Europe/Madrid"))
    with patch.object(sdl, "last_expected_market_date",
                      return_value=idx[-1].date()), \
         patch.object(sdl, "last_expected_lse_session",
                      return_value=idx[-1].date()):
        out = sdl.download_stock_prices(reference_date=ref)
    assert out is not None
    assert out.shape == df.shape


def test_download_cache_caducado_fuerza_descarga(tmp_path, monkeypatch):
    """Cache > CACHE_HOURS -> fuerza descarga y llama a get_stock_list."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    idx = pd.date_range("2026-09-20", periods=5, freq="D")
    cols = pd.MultiIndex.from_product([["Close"], ["AAA"]], names=["field", "ticker"])
    df = pd.DataFrame(np.ones((5, 1)), index=idx, columns=cols)
    pq_path = tmp_path / "data" / "stock_prices.parquet"
    df.to_parquet(pq_path)
    # Forzar mtime antiguo
    old = (datetime.now() - timedelta(hours=48)).timestamp()
    os.utime(pq_path, (old, old))

    with patch.object(sdl, "get_stock_list", return_value=[]) as mock_sl:
        out = sdl.download_stock_prices(
            reference_date=datetime.now(ZoneInfo("Europe/Madrid")))
    mock_sl.assert_called_once()
    assert out is None

# =============================================================================
# download_stock_prices: rama de descarga completa con mocks fuertes
# =============================================================================

def _mock_yf_download(batch, **kwargs):
    """Simula yf.download devolviendo un DataFrame MultiIndex valido."""
    idx = pd.date_range("2026-09-20", periods=5, freq="D")
    if isinstance(batch, str):
        batch = [batch]
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], batch],
        names=["field", "ticker"],
    )
    return pd.DataFrame(np.ones((5, len(cols))), index=idx, columns=cols)


def test_download_descarga_completa_minima(tmp_path, monkeypatch):
    """Sin cache, con 1 ticker Yahoo. Flujo: get_stock_list -> yf.download ->
    classify -> LSE override -> manifest. Verifica que devuelve DataFrame."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs" / "history").mkdir(parents=True)

    ref = datetime.now(ZoneInfo("Europe/Madrid"))

    with patch.object(sdl, "get_stock_list", return_value=["AAPL"]), \
         patch.object(sdl, "_get_yf_session", return_value=None), \
         patch.object(sdl, "last_expected_market_date",
                      return_value=pd.Timestamp("2026-09-24").date()), \
         patch.object(sdl, "last_expected_lse_session",
                      return_value=pd.Timestamp("2026-09-24").date()), \
         patch.object(sdl, "_apply_lse_close_override",
                      return_value=(_mock_yf_download(["AAPL"]), {})), \
         patch.object(sdl, "_classify_ticker",
                      return_value=("OK", None)), \
         patch("yfinance.download", side_effect=_mock_yf_download), \
         patch("src.utils.write_artifact_with_manifest", return_value={}):
        out = sdl.download_stock_prices(reference_date=ref)
    assert out is not None
    assert not out.empty
    assert ("Close", "AAPL") in out.columns


def test_download_lote_vacio_marca_failed(tmp_path, monkeypatch):
    """Si yf.download devuelve vacio, los tickers van a FAILED."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs" / "history").mkdir(parents=True)

    ref = datetime.now(ZoneInfo("Europe/Madrid"))

    with patch.object(sdl, "get_stock_list", return_value=["AAPL"]), \
         patch.object(sdl, "_get_yf_session", return_value=None), \
         patch.object(sdl, "last_expected_market_date",
                      return_value=pd.Timestamp("2026-09-24").date()), \
         patch.object(sdl, "last_expected_lse_session",
                      return_value=pd.Timestamp("2026-09-24").date()), \
         patch("yfinance.download", return_value=pd.DataFrame()):
        out = sdl.download_stock_prices(reference_date=ref)
    assert out is None


def test_download_cascade_europea_cubre_tickers(tmp_path, monkeypatch):
    """Con tickers europeos: se descargan via Euronext/Xetra/BME (no Yahoo)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "outputs" / "history").mkdir(parents=True)

    ref = datetime.now(ZoneInfo("Europe/Madrid"))

    def _fake_euronext_get(self, tickers, **kwargs):
        return _mock_yf_download(list(tickers))

    with patch.object(sdl, "get_stock_list", return_value=["AIR.PA"]), \
         patch.object(sdl.EuronextProvider, "supports", return_value=True), \
         patch.object(sdl.XetraProvider, "supports", return_value=False), \
         patch.object(sdl.BMEProvider, "supports", return_value=False), \
         patch.object(sdl.EuronextProvider, "get_prices", _fake_euronext_get), \
         patch.object(sdl, "_get_yf_session", return_value=None), \
         patch.object(sdl, "last_expected_market_date",
                      return_value=pd.Timestamp("2026-09-24").date()), \
         patch.object(sdl, "last_expected_lse_session",
                      return_value=pd.Timestamp("2026-09-24").date()), \
         patch.object(sdl, "_apply_lse_close_override",
                      return_value=(_mock_yf_download(["AIR.PA"]), {})), \
         patch("src.utils.write_artifact_with_manifest", return_value={}):
        out = sdl.download_stock_prices(reference_date=ref)
    assert out is not None
    assert ("Close", "AIR.PA") in out.columns