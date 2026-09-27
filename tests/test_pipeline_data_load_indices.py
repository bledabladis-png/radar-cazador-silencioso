"""Tests pipeline: data_load (fase 1) + indices_intl (fase 10b).

Orquestadores con mocks. Verifican contrato y ramas de degradacion.

Nota: docstring de compute_indices_intl menciona 3 keys pero devuelve 2
(index_phases, index_leaders). El test verifica el contrato REAL.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.data_load import load_all_data
from src.pipeline.indices_intl import compute_indices_intl


# =============================================================================
# data_load.py
# =============================================================================

def _df_market(n_rows=5):
    idx = pd.date_range("2026-09-20", periods=n_rows, freq="D")
    cols = pd.MultiIndex.from_product([["Close"], ["AAA"]], names=["field", "ticker"])
    return pd.DataFrame([1.0] * n_rows, index=idx, columns=cols)


def test_load_all_data_contrato_5_keys():
    with patch("src.pipeline.data_load.download_market_data", return_value=_df_market()), \
         patch("src.pipeline.data_load.validate_market_data",
               return_value=(["AAA", "BBB", "CCC", "DDD", "EEE"], {})), \
         patch("src.pipeline.data_load.load_macro_manual",
               return_value=pd.DataFrame({"date": [1, 2]})):
        out = load_all_data()
    assert set(out.keys()) == {
        "df_market", "df_macro_manual", "valid_tickers", "issues", "temporal_meta"}


def test_load_all_data_df_market_none_devuelve_none():
    with patch("src.pipeline.data_load.download_market_data", return_value=None):
        out = load_all_data()
    assert out is None


def test_load_all_data_df_market_empty_devuelve_none():
    with patch("src.pipeline.data_load.download_market_data", return_value=pd.DataFrame()):
        out = load_all_data()
    assert out is None


def test_load_all_data_pocos_tickers_devuelve_none():
    """len(valid) < 5 -> None."""
    with patch("src.pipeline.data_load.download_market_data", return_value=_df_market()), \
         patch("src.pipeline.data_load.validate_market_data",
               return_value=(["AAA", "BBB", "CCC"], {})):
        out = load_all_data()
    assert out is None


def test_load_all_data_imprime_issues():
    with patch("src.pipeline.data_load.download_market_data", return_value=_df_market()), \
         patch("src.pipeline.data_load.validate_market_data",
               return_value=(list("ABCDE"), {"XXX": "sin datos"})), \
         patch("src.pipeline.data_load.load_macro_manual", return_value=None):
        out = load_all_data()
    assert out["issues"] == {"XXX": "sin datos"}
    assert out["df_macro_manual"] is None


def test_load_all_data_propaga_reference_date_y_run_id():
    captured = {}
    def _fake_download(reference_date=None, run_id=None):
        captured["ref"] = reference_date
        captured["rid"] = run_id
        return _df_market()
    with patch("src.pipeline.data_load.download_market_data", side_effect=_fake_download), \
         patch("src.pipeline.data_load.validate_market_data",
               return_value=(list("ABCDE"), {})), \
         patch("src.pipeline.data_load.load_macro_manual", return_value=None):
        load_all_data(reference_date="2026-09-25", run_id="rid1")
    assert captured["ref"] == "2026-09-25"
    assert captured["rid"] == "rid1"


def test_load_all_data_temporal_meta_de_attrs():
    """temporal_meta viene de df_market.attrs."""
    df = _df_market()
    df.attrs["temporal_meta"] = {"by_contract": {"EQUITY_EOD": {"coverage": 0.95}}}
    with patch("src.pipeline.data_load.download_market_data", return_value=df), \
         patch("src.pipeline.data_load.validate_market_data",
               return_value=(list("ABCDE"), {})), \
         patch("src.pipeline.data_load.load_macro_manual", return_value=None):
        out = load_all_data()
    assert out["temporal_meta"] == {"by_contract": {"EQUITY_EOD": {"coverage": 0.95}}}


# =============================================================================
# indices_intl.py
# =============================================================================

def test_indices_intl_contrato_2_keys():
    with patch("src.pipeline.indices_intl.compute_index_phases",
               return_value=({}, {})), \
         patch("src.pipeline.indices_intl.download_stock_prices",
               return_value=None):
        out = compute_indices_intl(pd.DataFrame())
    assert set(out.keys()) == {"index_phases", "index_leaders"}


def test_indices_intl_sin_acumulacion_no_descarga():
    """Si ningun indice esta en ACCUMULATION/MARKUP, NO se descargan stocks."""
    with patch("src.pipeline.indices_intl.compute_index_phases",
               return_value=({"^GSPC": "RANGE", "^DJI": "DISTRIBUTION"}, {})), \
         patch("src.pipeline.indices_intl.download_stock_prices") as mock_dl:
        out = compute_indices_intl(pd.DataFrame())
    mock_dl.assert_not_called()
    assert out["index_leaders"] == {}


def test_indices_intl_con_acumulacion_descarga_y_selecciona(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "report").mkdir(parents=True)
    leaders_df = pd.DataFrame({"ticker": ["AAA"], "wls": [1.5]})
    with patch("src.pipeline.indices_intl.compute_index_phases",
               return_value=({"^DJI": "ACCUMULATION"}, {})), \
         patch("src.pipeline.indices_intl.download_stock_prices",
               return_value=pd.DataFrame({"dummy": [1]})), \
         patch("src.pipeline.indices_intl.select_index_leaders",
               return_value={"^DJI": leaders_df}):
        out = compute_indices_intl(pd.DataFrame())
    assert "^DJI" in out["index_leaders"]


def test_indices_intl_errores_por_indice_no_abortan():
    """Error al calcular lideres de un indice -> se loguea y continua."""
    def _fake_select(*args, **kwargs):
        raise RuntimeError("boom")
    with patch("src.pipeline.indices_intl.compute_index_phases",
               return_value=({"^DJI": "ACCUMULATION"}, {})), \
         patch("src.pipeline.indices_intl.download_stock_prices",
               return_value=pd.DataFrame()), \
         patch("src.pipeline.indices_intl.select_index_leaders",
               side_effect=_fake_select):
        out = compute_indices_intl(pd.DataFrame())
    assert out["index_leaders"] == {}


def test_indices_intl_csv_falla_no_rompe(tmp_path, monkeypatch):
    """Error al escribir CSV de lideres no aborta."""
    monkeypatch.chdir(tmp_path)
    leaders_df = pd.DataFrame({"ticker": ["AAA"]})
    with patch("src.pipeline.indices_intl.compute_index_phases",
               return_value=({"^DJI": "ACCUMULATION"}, {})), \
         patch("src.pipeline.indices_intl.download_stock_prices",
               return_value=pd.DataFrame()), \
         patch("src.pipeline.indices_intl.select_index_leaders",
               return_value={"^DJI": leaders_df}), \
         patch.object(pd.DataFrame, "to_csv", side_effect=RuntimeError("disk full")):
        # No debe propagar
        out = compute_indices_intl(pd.DataFrame())
    assert "^DJI" in out["index_leaders"]


def test_indices_intl_csv_escrito(tmp_path, monkeypatch):
    """Con lideres validos, escribe CSV en outputs/report/."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "report").mkdir(parents=True)
    leaders_df = pd.DataFrame({"ticker": ["AAA", "BBB"], "wls": [1.5, 1.0]})
    with patch("src.pipeline.indices_intl.compute_index_phases",
               return_value=({"^DJI": "ACCUMULATION"}, {})), \
         patch("src.pipeline.indices_intl.download_stock_prices",
               return_value=pd.DataFrame()), \
         patch("src.pipeline.indices_intl.select_index_leaders",
               return_value={"^DJI": leaders_df}):
        compute_indices_intl(pd.DataFrame())
    csv_path = tmp_path / "outputs" / "report" / "analisis_lideres_internacionales.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert "indice" in df.columns
    assert df.iloc[0]["indice"] == "^DJI"