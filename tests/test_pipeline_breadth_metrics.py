"""Tests pipeline: breadth_metrics (fase 6c).

Cubre _compute_momentum_amplitud, _load_latest_valid_breadth_snapshot,
_compute_sector_breadth_health y compute_breadth_metrics.

Logica clave:
  - B2: no se genera nueva observacion si el dia no es sesion NYSE
    o si la sesion esperada no esta en df_stocks.
  - C2F: fallback al ultimo snapshot valido con is_stale=True.
"""
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import breadth_metrics as bm


# NYSE market days de referencia (verificados en el calendario del sistema)
MARKET_DAY = datetime(2026, 9, 25)       # viernes
NON_MARKET_DAY = datetime(2026, 9, 26)   # sabado


def _df_stocks(n=5):
    idx = pd.date_range("2026-09-20", periods=n, freq="D")
    cols = pd.MultiIndex.from_product([["Close"], ["AAA"]], names=["field", "ticker"])
    return pd.DataFrame([1.0] * n, index=idx, columns=cols)


def _df_stocks_hasta(dias_max):
    """df_stocks cuyo index termina en `dias_max` (date)."""
    idx = pd.date_range(end=pd.Timestamp(dias_max), periods=5, freq="D")
    cols = pd.MultiIndex.from_product([["Close"], ["AAA"]], names=["field", "ticker"])
    return pd.DataFrame([1.0] * 5, index=idx, columns=cols)


# =============================================================================
# _compute_momentum_amplitud
# =============================================================================

def test_momentum_amplitud_df_stocks_none():
    out = bm._compute_momentum_amplitud(None)
    assert out is None


def test_momentum_amplitud_df_stocks_empty():
    out = bm._compute_momentum_amplitud(pd.DataFrame())
    assert out is None


def test_momentum_amplitud_calcula_y_persiste(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    fake_sbm = pd.DataFrame([{"date": "2026-09-25", "sector": "XLK",
                              "delta_1d_ema20": 0.1}])
    with patch("src.pipeline.breadth_metrics.compute_sector_breadth_momentum",
               return_value=fake_sbm):
        out = bm._compute_momentum_amplitud(_df_stocks())
    assert out is not None
    p = tmp_path / "outputs" / "history" / "sector_breadth_momentum.csv"
    assert p.exists()


def test_momentum_amplitud_empty_result_devuelve_df_vacio(tmp_path, monkeypatch):
    """Si compute_sector_breadth_momentum devuelve df vacio, la funcion
    lo propaga (no None). El CSV NO se persiste."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    with patch("src.pipeline.breadth_metrics.compute_sector_breadth_momentum",
               return_value=pd.DataFrame()):
        out = bm._compute_momentum_amplitud(_df_stocks())
    assert out is not None
    assert out.empty
    # No se persiste CSV con df vacio
    p = tmp_path / "outputs" / "history" / "sector_breadth_momentum.csv"
    assert not p.exists()


def test_momentum_amplitud_error_devuelve_none():
    with patch("src.pipeline.breadth_metrics.compute_sector_breadth_momentum",
               side_effect=RuntimeError("x")):
        out = bm._compute_momentum_amplitud(_df_stocks())
    assert out is None


# =============================================================================
# _load_latest_valid_breadth_snapshot
# =============================================================================

def test_snapshot_no_existe(tmp_path):
    out = bm._load_latest_valid_breadth_snapshot(tmp_path / "no_existe.csv")
    assert out is None


def test_snapshot_csv_vacio(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("date,sector\n", encoding="utf-8")
    assert bm._load_latest_valid_breadth_snapshot(p) is None


def test_snapshot_sin_columna_date(tmp_path):
    p = tmp_path / "no_date.csv"
    p.write_text("foo,bar\n1,2\n", encoding="utf-8")
    assert bm._load_latest_valid_breadth_snapshot(p) is None


def test_snapshot_dias_no_bursatiles_filtrados(tmp_path):
    """Un sabado con 11 sectores -> filtrado, sin snapshot valido."""
    p = tmp_path / "solo_sabado.csv"
    filas = ["2026-09-26,XLK"] + [f"2026-09-26,{s}" for s in
        ["XLF","XLV","XLE","XLY","XLP","XLI","XLB","XLU","XLRE","XLC"]]
    p.write_text("date,sector\n" + "\n".join(filas), encoding="utf-8")
    assert bm._load_latest_valid_breadth_snapshot(p) is None


def test_snapshot_dia_mercado_insuficientes_sectores(tmp_path):
    """Viernes con solo 3 sectores -> no valido."""
    p = tmp_path / "pocos.csv"
    filas = ["2026-09-25,XLK", "2026-09-25,XLF", "2026-09-25,XLV"]
    p.write_text("date,sector\n" + "\n".join(filas), encoding="utf-8")
    assert bm._load_latest_valid_breadth_snapshot(p) is None


def test_snapshot_valido(tmp_path):
    """Viernes con 11 sectores -> devuelve snapshot."""
    p = tmp_path / "ok.csv"
    sectores = ["XLK","XLF","XLV","XLE","XLY","XLP","XLI","XLB","XLU","XLRE","XLC"]
    filas = [f"2026-09-25,{s}" for s in sectores]
    p.write_text("date,sector\n" + "\n".join(filas), encoding="utf-8")
    out = bm._load_latest_valid_breadth_snapshot(p)
    assert out is not None
    assert len(out) == 11


# =============================================================================
# _compute_sector_breadth_health
# =============================================================================

def test_sbh_df_stocks_none():
    out, is_stale = bm._compute_sector_breadth_health(
        None, pd.DataFrame(), pd.DataFrame())
    assert out is None
    assert is_stale is False


def test_sbh_df_stocks_vacio():
    out, is_stale = bm._compute_sector_breadth_health(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert out is None
    assert is_stale is False


def test_sbh_dia_no_bursatil_devuelve_stale(tmp_path, monkeypatch):
    """Sabado -> is_stale=True (sin nueva observacion)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    out, is_stale = bm._compute_sector_breadth_health(
        _df_stocks(), pd.DataFrame(), pd.DataFrame(),
        reference_date=NON_MARKET_DAY,
        output_path=tmp_path / "outputs" / "history" / "sb.csv",
    )
    assert is_stale is True
    # Sin snapshot previo -> out=None
    assert out is None


def test_sbh_expected_session_ausente_devuelve_stale(tmp_path, monkeypatch):
    """df_stocks termina antes de la sesion esperada -> is_stale=True."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    df_stocks = _df_stocks_hasta("2026-09-20")
    with patch("src.pipeline.breadth_metrics.last_expected_market_date",
               return_value=pd.Timestamp("2026-09-25").date()):
        out, is_stale = bm._compute_sector_breadth_health(
            df_stocks, pd.DataFrame(), pd.DataFrame(),
            reference_date=MARKET_DAY,
            output_path=tmp_path / "outputs" / "history" / "sb.csv",
        )
    assert is_stale is True


def test_sbh_dia_mercado_calcula_observacion(tmp_path, monkeypatch):
    """Viernes con df_stocks hasta la sesion esperada -> observacion nueva."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    df_stocks = _df_stocks_hasta("2026-09-25")
    fake_sb = pd.DataFrame([
        {"date": pd.Timestamp("2026-09-25"), "sector": "XLK",
         "n_total": 20, "n_valid_ema200": 20},
    ])
    with patch("src.pipeline.breadth_metrics.last_expected_market_date",
               return_value=pd.Timestamp("2026-09-25").date()), \
         patch("src.pipeline.breadth_metrics.compute_sector_breadth",
               return_value=fake_sb):
        out, is_stale = bm._compute_sector_breadth_health(
            df_stocks, pd.DataFrame(), pd.DataFrame(),
            reference_date=MARKET_DAY,
            output_path=tmp_path / "outputs" / "history" / "sb.csv",
        )
    assert is_stale is False
    assert out is not None
    assert len(out) == 1


def test_sbh_error_en_compute_devuelve_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    df_stocks = _df_stocks_hasta("2026-09-25")
    with patch("src.pipeline.breadth_metrics.last_expected_market_date",
               return_value=pd.Timestamp("2026-09-25").date()), \
         patch("src.pipeline.breadth_metrics.compute_sector_breadth",
               side_effect=RuntimeError("x")):
        out, is_stale = bm._compute_sector_breadth_health(
            df_stocks, pd.DataFrame(), pd.DataFrame(),
            reference_date=MARKET_DAY,
            output_path=tmp_path / "outputs" / "history" / "sb.csv",
        )
    assert out is None
    assert is_stale is False


# =============================================================================
# compute_breadth_metrics
# =============================================================================

def test_compute_breadth_metrics_contrato_4_keys(tmp_path, monkeypatch):
    """F6-2b (2026-09-28): contrato ampliado a 4 keys.

    sector_breadth_stale_reason se anade para distinguir
    MARKET_CLOSED vs DATA_PENDING (F6-01).
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    with patch("src.pipeline.breadth_metrics._compute_sector_breadth_health",
               return_value=(None, True, 'DATA_PENDING')), \
         patch("src.pipeline.breadth_metrics._compute_momentum_amplitud",
               return_value=None):
        out = bm.compute_breadth_metrics(
            df_stocks=pd.DataFrame(), df_market=pd.DataFrame(),
            holdings_df=pd.DataFrame(),
        )
    assert set(out.keys()) == {
        "sector_breadth_momentum_df", "sector_breadth_df",
        "sector_breadth_is_stale", "sector_breadth_stale_reason"}


def test_compute_breadth_metrics_propaga_is_stale(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    fake_sb = pd.DataFrame([{"date": "2026-09-24", "sector": "XLK"}])
    with patch("src.pipeline.breadth_metrics._compute_sector_breadth_health",
               return_value=(fake_sb, True, 'MARKET_CLOSED')), \
         patch("src.pipeline.breadth_metrics._compute_momentum_amplitud",
               return_value=None):
        out = bm.compute_breadth_metrics(
            df_stocks=pd.DataFrame(), df_market=pd.DataFrame(),
            holdings_df=pd.DataFrame(),
        )
    assert out["sector_breadth_is_stale"] is True
    assert out["sector_breadth_stale_reason"] == 'MARKET_CLOSED'
    assert out["sector_breadth_df"] is fake_sb


def test_momentum_amplitud_fallo_to_csv_no_pierde_filas(tmp_path, monkeypatch):
    """Si to_csv muere a mitad, el CSV historico debe quedar intacto."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    csv = tmp_path / "outputs" / "history" / "sector_breadth_momentum.csv"
    pd.DataFrame([{"date": "2026-09-24", "sector": "XLK",
                   "delta_1d_ema20": 0.05}]).to_csv(csv, index=False)
    filas_antes = pd.read_csv(csv).shape[0]
    assert filas_antes == 1

    real_to_csv = pd.DataFrame.to_csv

    def fake_to_csv(self, path, **kw):
        real_to_csv(self.iloc[:0], path, **kw)
        raise OSError("simulado: fallo a mitad de escritura")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)
    fake_sbm = pd.DataFrame([{"date": "2026-09-25", "sector": "XLK",
                              "delta_1d_ema20": 0.1}])
    with patch("src.pipeline.breadth_metrics.compute_sector_breadth_momentum",
               return_value=fake_sbm):
        bm._compute_momentum_amplitud(_df_stocks())

    df = pd.read_csv(csv)
    assert df.shape[0] == filas_antes, (
        f"CSV original perdio filas: {filas_antes} -> {df.shape[0]}"
    )
    assert df.iloc[0]["date"] == "2026-09-24"


def test_sbh_fallo_to_csv_no_pierde_filas(tmp_path, monkeypatch):
    """Si to_csv muere a mitad, el CSV historico debe quedar intacto."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    csv = tmp_path / "outputs" / "history" / "sb.csv"
    pd.DataFrame([{"date": "2026-09-24", "sector": "XLK",
                   "n_total": 20, "n_valid_ema200": 20}]).to_csv(csv, index=False)
    filas_antes = pd.read_csv(csv).shape[0]
    assert filas_antes == 1

    real_to_csv = pd.DataFrame.to_csv

    def fake_to_csv(self, path, **kw):
        real_to_csv(self.iloc[:0], path, **kw)
        raise OSError("simulado: fallo a mitad de escritura")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)
    df_stocks = _df_stocks_hasta("2026-09-25")
    fake_sb = pd.DataFrame([
        {"date": pd.Timestamp("2026-09-25"), "sector": "XLK",
         "n_total": 20, "n_valid_ema200": 20},
    ])
    with patch("src.pipeline.breadth_metrics.last_expected_market_date",
               return_value=pd.Timestamp("2026-09-25").date()), \
         patch("src.pipeline.breadth_metrics.compute_sector_breadth",
               return_value=fake_sb):
        bm._compute_sector_breadth_health(
            df_stocks, pd.DataFrame(), pd.DataFrame(),
            reference_date=MARKET_DAY,
            output_path=csv,
        )

    df = pd.read_csv(csv)
    assert df.shape[0] == filas_antes, (
        f"CSV original perdio filas: {filas_antes} -> {df.shape[0]}"
    )
    assert df.iloc[0]["date"] == "2026-09-24"