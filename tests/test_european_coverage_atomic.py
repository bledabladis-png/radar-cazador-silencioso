"""Test del bug real de atomicidad en european_coverage._append_csv.

Familia 3 (2026-09-29): el CSV es historico append+dedup. Si el
proceso muere a mitad de to_csv, el fichero queda truncado y el
siguiente run lee menos filas -> perdidas irrecuperables.

Este test simula el fallo y verifica que el CSV original queda
INTACTO. Verifica el bug, no el patron.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.european_coverage import _append_csv


def _rows():
    return [{
        "ticker": "AIR.PA",
        "source": "Euronext",
        "last_date": None,
        "days_gap": 0,
        "status": "OK",
    }]


def test_append_csv_fallo_de_to_csv_no_pierde_filas(tmp_path, monkeypatch):
    """Simula to_csv que muere a mitad. El CSV original debe quedar intacto."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    csv = tmp_path / "outputs" / "history" / "european_coverage.csv"

    # Estado inicial: 1 fila historica, escrita sin monkeypatch.
    pd.DataFrame([{
        "date": "2026-09-28",
        "ticker": "AIR.PA",
        "source": "Euronext",
        "last_date": "2026-09-25",
        "days_gap": 3,
        "status": "OK",
    }]).to_csv(csv, index=False)
    filas_antes = pd.read_csv(csv).shape[0]
    assert filas_antes == 1

    # Simular: to_csv trunca el destino y luego revienta.
    real_to_csv = pd.DataFrame.to_csv

    def fake_to_csv(self, path, **kw):
        real_to_csv(self.iloc[:0], path, **kw)  # escribe 0 filas al destino
        raise OSError("simulado: disco lleno a mitad de escritura")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)

    with pytest.raises(OSError, match="simulado"):
        _append_csv(_rows(), "2026-09-29")

    # El CSV original debe seguir con sus filas.
    df_despues = pd.read_csv(csv)
    assert df_despues.shape[0] == filas_antes, (
        f"El CSV original perdio filas: {filas_antes} -> {df_despues.shape[0]}"
    )
    assert df_despues.iloc[0]["date"] == "2026-09-28"