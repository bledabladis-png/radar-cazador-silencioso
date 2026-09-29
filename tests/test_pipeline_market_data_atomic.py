"""Test del bug real de atomicidad en src/pipeline/market_data.py."""
from __future__ import annotations

import pandas as pd
from unittest.mock import patch

from src.pipeline import market_data as md


def _fake_to_csv_that_fails():
    real_to_csv = pd.DataFrame.to_csv

    def fake_to_csv(self, path, **kw):
        real_to_csv(self.iloc[:0], path, **kw)
        raise OSError("simulado: fallo a mitad de escritura")

    return fake_to_csv


def test_vol_structure_fallo_to_csv_no_pierde_filas(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    csv = tmp_path / "outputs" / "history" / "volatility_structure.csv"
    pd.DataFrame([{"date": "2026-09-24", "vix_z": 0.1}]).to_csv(csv, index=False)
    filas_antes = pd.read_csv(csv).shape[0]
    assert filas_antes == 1

    fake_df = pd.DataFrame([{"date": "2026-09-25", "vix_z": 0.2}])
    with patch("indicators.volatility_structure.compute_volatility_structure",
               return_value=fake_df), \
         patch.object(pd.DataFrame, "to_csv", _fake_to_csv_that_fails()):
        md._compute_vol_structure(pd.DataFrame(), None)

    df = pd.read_csv(csv)
    assert df.shape[0] == filas_antes, (
        f"CSV original perdio filas: {filas_antes} -> {df.shape[0]}"
    )
    assert df.iloc[0]["date"] == "2026-09-24"


def test_data_quality_fallo_to_csv_no_pierde_filas(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    csv = tmp_path / "outputs" / "history" / "data_quality.csv"
    pd.DataFrame([{"date": "2026-09-24", "source": "X", "status": "OK"}]).to_csv(csv, index=False)
    filas_antes = pd.read_csv(csv).shape[0]
    assert filas_antes == 1

    fake_df = pd.DataFrame([{"date": "2026-09-25", "source": "X", "status": "OK"}])
    with patch("indicators.data_quality.compute_data_quality",
               return_value=fake_df), \
         patch.object(pd.DataFrame, "to_csv", _fake_to_csv_that_fails()):
        md._compute_data_quality()

    df = pd.read_csv(csv)
    assert df.shape[0] == filas_antes, (
        f"CSV original perdio filas: {filas_antes} -> {df.shape[0]}"
    )
    assert df.iloc[0]["date"] == "2026-09-24"