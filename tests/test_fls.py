# -*- coding: utf-8 -*-
"""D25 (2026-09-30): tests de indicators/fls.py.

Cubre:
- _zscore_last_over_lookback: funcion pura. Casos <252, MAD=0,
  ultimo valor no finito (guard finitud 2026-09-29), caso normal.
- compute_fls: lee 5 CSV de data/macro_manual. Test con chdir a
  tmp_path y CSVs sinteticos. Todos presentes, alguno ausente,
  todos ausentes, stressed_count, normalizacion.
"""
import numpy as np
import pandas as pd

from indicators.fls import _zscore_last_over_lookback, compute_fls


# ---------- _zscore_last_over_lookback ----------
def test_zscore_serie_corta_devuelve_cero():
    """<window valores no-NaN -> 0.0."""
    s = pd.Series([1.0] * 100)
    assert _zscore_last_over_lookback(s, window=252) == 0.0


def test_zscore_serie_constante_mad_cero():
    """Serie constante -> MAD=0 -> 0.0."""
    s = pd.Series([5.0] * 300)
    assert _zscore_last_over_lookback(s, window=252) == 0.0


def test_zscore_ultimo_nan_devuelve_cero():
    """Guard finitud 2026-09-29: ultimo valor NaN -> 0.0."""
    vals = list(np.random.RandomState(42).randn(300))
    vals[-1] = float("nan")
    s = pd.Series(vals)
    assert _zscore_last_over_lookback(s, window=252) == 0.0


def test_zscore_ultimo_inf_devuelve_cero():
    """Guard finitud: ultimo valor inf -> 0.0."""
    vals = list(np.random.RandomState(42).randn(300))
    vals[-1] = float("inf")
    s = pd.Series(vals)
    assert _zscore_last_over_lookback(s, window=252) == 0.0


def test_zscore_outlier_positivo():
    """Ultima observacion muy por encima de la mediana -> z positivo."""
    rng = np.random.RandomState(42)
    vals = list(rng.randn(252)) + [10.0]
    s = pd.Series(vals)
    z = _zscore_last_over_lookback(s, window=252)
    assert z > 1.0


def test_zscore_outlier_negativo():
    rng = np.random.RandomState(42)
    vals = list(rng.randn(252)) + [-10.0]
    s = pd.Series(vals)
    z = _zscore_last_over_lookback(s, window=252)
    assert z < -1.0


def test_zscore_ultimo_igual_a_mediana():
    """Ultimo valor == mediana -> z ~ 0."""
    # El ultimo es max, no mediana. Forzamos ultimo = mediana.
    s2 = pd.Series(list(np.linspace(0, 1, 300)[:-1]) + [0.5])
    z2 = _zscore_last_over_lookback(s2, window=252)
    assert abs(z2) < 0.5


def test_zscore_ignora_nan_no_ultimos():
    """NaN en medio de la serie no afectan (dropna de la ventana)."""
    rng = np.random.RandomState(42)
    vals = list(rng.randn(300))
    vals[150] = float("nan")
    vals[200] = float("nan")
    s = pd.Series(vals)
    z = _zscore_last_over_lookback(s, window=252)
    assert np.isfinite(z)


# ---------- compute_fls ----------
def _write_fls_csv(base, name, col, values):
    """Escribe <base>/<name> con <col>=values."""
    d = base / "data" / "macro_manual"
    d.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
    pd.DataFrame({col: values}, index=idx).to_csv(
        d / name, index=True)


def _write_all_fls_csvs(base, n=300):
    """5 CSVs con datos monotonos crecientes (z distinto de 0)."""
    _write_fls_csv(base, "sofr.csv", "SOFR",
                   np.linspace(1.0, 5.0, n))
    _write_fls_csv(base, "walcl.csv", "WALCL",
                   np.linspace(7e6, 8e6, n))
    _write_fls_csv(base, "rrpp.csv", "RRPONTSYD",
                   np.linspace(2e6, 1e6, n))
    _write_fls_csv(base, "commercial_paper.csv", "COMPOUT",
                   np.linspace(1e6, 1.2e6, n))
    _write_fls_csv(base, "discount_rate.csv", "DPRIME",
                   np.linspace(5.0, 5.5, n))


def test_compute_fls_todos_presentes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_all_fls_csvs(tmp_path)
    out = compute_fls()
    assert out["components"] == 5
    assert out["total_components"] == 5
    assert 0.0 <= out["fls_value"] <= 1.0
    assert 0.0 <= out["fls_normalized"] <= 1.0
    for k in ("SOFR", "WALCL", "RRP", "CP", "Discount"):
        assert k in out["detail"]
        assert out["detail"][k]["value"] is not None


def test_compute_fls_sin_csvs(tmp_path, monkeypatch):
    """Sin CSVs -> stresses vacio -> fallback."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "macro_manual").mkdir(parents=True)
    out = compute_fls()
    assert out["components"] == 0
    assert out["fls_value"] == 0.0
    assert out["fls_normalized"] == 0.5
    assert out["stressed_components"] == 0


def test_compute_fls_csv_parcial(tmp_path, monkeypatch):
    """Solo SOFR presente -> 1 componente, los otros None."""
    monkeypatch.chdir(tmp_path)
    _write_fls_csv(tmp_path, "sofr.csv", "SOFR",
                   np.linspace(1.0, 5.0, 300))
    out = compute_fls()
    assert out["components"] == 1
    assert out["detail"]["SOFR"]["value"] is not None
    assert out["detail"]["WALCL"]["value"] is None


def test_compute_fls_stressed_count(tmp_path, monkeypatch):
    """Componentes con z tanh > 0.3 se cuentan como stressed."""
    monkeypatch.chdir(tmp_path)
    # Todos monotonicamente crecientes -> ultimo es max, z alto
    _write_all_fls_csvs(tmp_path)
    out = compute_fls()
    assert 0 <= out["stressed_components"] <= 5
    # Contar manualmente
    expected = sum(1 for d in out["detail"].values()
                   if d.get("stressed", False))
    assert out["stressed_components"] == expected


def test_compute_fls_normalized_rango(tmp_path, monkeypatch):
    """fls_normalized en [0, 1] siempre."""
    monkeypatch.chdir(tmp_path)
    _write_all_fls_csvs(tmp_path, n=300)
    out = compute_fls()
    assert 0.0 <= out["fls_normalized"] <= 1.0
