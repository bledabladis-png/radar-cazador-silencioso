"""Test escritura atomica en european_coverage._append_csv.

Familia 3 (2026-09-29): el CSV es historico append+dedup. Si el
proceso muere a mitad de to_csv, el fichero queda truncado y el
siguiente run lee menos filas. Fix: .tmp + Path.replace.

Patron: verificar que no quedan .tmp residuales tras escritura
exitosa (mismo que test_lse_scraper_loader:484 y
test_sec_13f_manifest:150).
"""
from __future__ import annotations

import pandas as pd

from src.european_coverage import _append_csv


def _setup_tmp(tmp_path, monkeypatch):
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)


def _rows(ticker="AIR.PA", source="Euronext", status="OK"):
    return [{
        "ticker": ticker,
        "source": source,
        "last_date": None,
        "days_gap": 0,
        "status": status,
    }]


def test_append_csv_sin_tmp_residual(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    _append_csv(_rows(), "2026-09-29")
    csv = tmp_path / "outputs" / "history" / "european_coverage.csv"
    assert csv.exists()
    tmps = list((tmp_path / "outputs" / "history").glob("*.tmp*"))
    assert tmps == [], f"quedan temporales: {tmps}"


def test_append_csv_preserva_filas_previas(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    csv = tmp_path / "outputs" / "history" / "european_coverage.csv"
    pd.DataFrame([{
        "date": "2026-09-28", "ticker": "AIR.PA", "source": "Euronext",
        "last_date": "2026-09-25", "days_gap": 3, "status": "OK",
    }]).to_csv(csv, index=False)

    _append_csv(_rows(), "2026-09-29")

    df = pd.read_csv(csv)
    assert len(df) == 2
    assert set(df["date"]) == {"2026-09-28", "2026-09-29"}


def test_append_csv_sin_rows_no_escribe(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    _append_csv([], "2026-09-29")
    csv = tmp_path / "outputs" / "history" / "european_coverage.csv"
    assert not csv.exists()