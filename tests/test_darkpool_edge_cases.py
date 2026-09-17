"""DT3 Fase 5: edge cases de indicators/darkpool (post-refactor).

Cubre comportamiento defensivo de las 4 funciones publicas y de los
modulos extraidos en Fases 2-4.

No depende de data/*.parquet.
"""
import numpy as np
import pandas as pd

from indicators.darkpool import (
    classify_darkpool,
    robust_zscore,
    rolling_percentile,
    _get_all_tickers,
    _get_volume_from_df,
    _compute_z_for_window,
)
from config.settings import DARKPOOL_THRESHOLDS


# --- robust_zscore ---

def test_robust_zscore_mad_cero_devuelve_ceros():
    s = pd.Series([5.0, 5.0, 5.0, 5.0])
    out = robust_zscore(s)
    assert (out == 0).all()


def test_robust_zscore_outlier_positivo():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    out = robust_zscore(s)
    assert out.iloc[-1] > 0


def test_robust_zscore_serie_vacia():
    # K-DT3-RUNTIMEWARN: ademas de len(out)==0, exigimos que no se emita
    # RuntimeWarning de numpy sobre la serie vacia.
    import warnings
    s = pd.Series([], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = robust_zscore(s)
    assert len(out) == 0


# --- rolling_percentile ---

def test_rolling_percentile_unico_elemento():
    s = pd.Series([7.0])
    assert rolling_percentile(s) == 0.0


def test_rolling_percentile_maximo():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rolling_percentile(s) == 80.0


# --- classify_darkpool ---

def test_classify_extremadamente_alta():
    z = DARKPOOL_THRESHOLDS["extremadamente_alta"]
    assert "extremadamente alta" in classify_darkpool(z)


def test_classify_extremadamente_baja():
    z = DARKPOOL_THRESHOLDS["muy_baja"] - 1.0
    assert "extremadamente baja" in classify_darkpool(z)


def test_classify_retorna_str():
    for z in [-5.0, -1.0, 0.0, 1.0, 5.0]:
        assert isinstance(classify_darkpool(z), str)


# --- _get_all_tickers ---

def test_get_all_tickers_sin_csv_no_rompe(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    tickers = _get_all_tickers()
    assert isinstance(tickers, list)


def test_get_all_tickers_filtra_formato_invalido(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "^VIX", "-BAD", "1NUMBER", "", "AB"]
    }).to_csv(tmp_path / "data" / "etf_holdings.csv", index=False)
    tickers = _get_all_tickers()
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "^VIX" not in tickers
    assert "-BAD" not in tickers
    assert "" not in tickers


# --- _get_volume_from_df ---

def test_get_volume_from_df_vacio():
    df = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples([("Volume", "AAPL")], names=["f", "t"])
    )
    out = _get_volume_from_df(df, "2026-01-01", "2026-01-07")
    assert out == {}


def test_get_volume_from_df_suma_en_rango():
    dates = pd.date_range("2026-01-01", periods=5, freq="B")
    df = pd.DataFrame(
        {("Volume", "AAPL"): [10.0, 20.0, 30.0, 40.0, 50.0]},
        index=dates,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    out = _get_volume_from_df(df, "2026-01-02", "2026-01-05")
    # 2026-01-02 (20.0) + 2026-01-05 (30.0) = 50.0
    assert out == {"AAPL": 50.0}


# --- _compute_z_for_window ---

def test_compute_z_for_window_hist_insuficiente():
    hist = pd.DataFrame({"week": pd.date_range("2026-01-01", periods=5, freq="W-MON"),
                         "ratio": [0.2] * 5})
    z, mom, pct, state = _compute_z_for_window(hist, 13)
    assert np.isnan(z)
    assert np.isnan(mom)
    assert np.isnan(pct)
    assert state == "Sin historial suficiente"


def test_compute_z_for_window_ok():
    n = 26
    rng = np.random.RandomState(7)
    hist = pd.DataFrame({
        "week": pd.date_range("2024-01-01", periods=n, freq="W-MON"),
        "ratio": rng.uniform(0.15, 0.30, n),
    })
    z, mom, pct, state = _compute_z_for_window(hist, 13)
    assert not np.isnan(z)
    assert not np.isnan(mom)
    assert 0.0 <= pct <= 100.0
    assert isinstance(state, str)


# --- Paquete refactorizado: import paths ---

def test_scoring_module_importable():
    import indicators.darkpool_scoring as m
    assert hasattr(m, "robust_zscore")
    assert hasattr(m, "classify_darkpool")


def test_io_module_importable():
    import indicators.darkpool_io as m
    assert hasattr(m, "_get_all_tickers")
    assert hasattr(m, "_get_volume_from_df")


def test_history_module_importable():
    import indicators.darkpool_history as m
    assert hasattr(m, "_backfill_history")


def test_reexports_identidad():
    """Los re-exports en darkpool.py apuntan a los modulos reales."""
    import indicators.darkpool as dp
    import indicators.darkpool_scoring as sc
    import indicators.darkpool_io as io
    import indicators.darkpool_history as hi
    assert dp.robust_zscore is sc.robust_zscore
    assert dp._get_all_tickers is io._get_all_tickers
    assert dp._backfill_history is hi._backfill_history