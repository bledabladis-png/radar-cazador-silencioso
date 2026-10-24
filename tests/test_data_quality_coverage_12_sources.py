# -*- coding: utf-8 -*-
"""Regresion H2 (2026-10-17): 12 fuentes con coverage_logic funcional.

Antes: 11 fuentes con `coverage_logic: None` en SOURCES -> el reporte
mostraba N/D en cobertura aunque el CSV tenia datos. El commit
1d3e0a17 (2026-10-10) declaro las fuentes pero no implemento la
logica.

El test verifica:
1. Estructural: cada una de las 12 fuentes declara coverage_logic
   valido y, si es 'rows', declara coverage_col.
2. Funcional: la logica de calculo produce coverage numerico sobre
   CSVs sinteticos (tmp_path), sin depender de outputs/ gitignored.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators import data_quality as dq


TARGETS = {
    "CFTC Position Flow": ("rows", "net_position"),
    "FINRA Dark Pool": ("rows", "ratio"),
    "Macro FRED": ("rows", "macro_score"),
    "RS Internal": ("sectorial", None),
    "SEC N-PORT (posiciones)": ("rows", "POSITION_CHANGE"),
    "SEC N-PORT-P QQQ (flujos)": ("rows", "net_flow"),
    "Sector Correlation Summary": ("from_row", None),
    "Sector Dispersion": ("from_row", None),
    "Sector Rank History": ("sectorial", None),
    "Volatility Structure": ("rows", "vix_level"),
}

VALID_LOGICS = {"sectorial", "rows", "from_row",
                "flow_tickers", "sectorial_family", "sectorial_evidence"}


def test_12_fuentes_declaran_coverage_logic():
    """Cada una de las 12 fuentes tiene coverage_logic != None."""
    by_source = {s["source"]: s for s in dq.SOURCES}
    missing = []
    for src_name, (expected_logic, expected_col) in TARGETS.items():
        s = by_source.get(src_name)
        if s is None:
            missing.append((src_name, "no esta en SOURCES"))
            continue
        logic = s.get("coverage_logic")
        if logic is None:
            missing.append((src_name, "coverage_logic=None"))
            continue
        if logic not in VALID_LOGICS:
            missing.append((src_name, f"logic invalida: {logic!r}"))
            continue
        if expected_logic == "rows":
            col = s.get("coverage_col")
            if not col:
                missing.append((src_name, "rows sin coverage_col"))
    assert not missing, f"Fuentes con coverage incompleto: {missing}"


def test_coverage_rows_sobre_csv_sintetico(tmp_path, monkeypatch):
    """logic='rows' + coverage_col: n_valid cuenta no-NaN."""
    csv_path = tmp_path / "test.csv"
    pd.DataFrame({
        "date": ["2026-10-14", "2026-10-15", "2026-10-16"],
        "net_position": [100.0, np.nan, 200.0],
    }).to_csv(csv_path, index=False)

    src = {
        "source": "T", "file": str(csv_path), "date_cols": ["date"],
        "frequency": "daily", "coverage_logic": "rows",
        "coverage_col": "net_position", "notes": "",
    }
    monkeypatch.setattr(dq, "SOURCES", [src])
    df = dq.compute_data_quality(reference_date=pd.Timestamp("2026-10-16"))
    row = df[df["source"] == "T"].iloc[-1]
    assert row["n_total"] == 3
    assert row["n_valid"] == 2
    assert row["coverage"] == pytest.approx(2.0 / 3.0)


def test_coverage_sectorial_sobre_csv_sintetico(tmp_path, monkeypatch):
    """logic='sectorial': n_valid = sectores unicos en ultima fecha."""
    csv_path = tmp_path / "test.csv"
    pd.DataFrame({
        "date": ["2026-10-16"] * 11,
        "sector": ["XLK", "XLF", "XLV", "XLE", "XLY", "XLP",
                   "XLI", "XLB", "XLU", "XLRE", "XLC"],
    }).to_csv(csv_path, index=False)

    src = {
        "source": "T", "file": str(csv_path), "date_cols": ["date"],
        "frequency": "daily", "coverage_logic": "sectorial", "notes": "",
    }
    monkeypatch.setattr(dq, "SOURCES", [src])
    df = dq.compute_data_quality(reference_date=pd.Timestamp("2026-10-16"))
    row = df[df["source"] == "T"].iloc[-1]
    assert row["n_total"] == 11
    assert row["n_valid"] == 11
    assert row["coverage"] == pytest.approx(1.0)


def test_coverage_from_row_sobre_csv_sintetico(tmp_path, monkeypatch):
    """logic='from_row': lee n_total/n_valid/coverage de la fila."""
    csv_path = tmp_path / "test.csv"
    pd.DataFrame({
        "date": ["2026-10-16"],
        "n_sectors": [11],
        "n_valid_pairs": [55],
        "coverage": [0.9944],
    }).to_csv(csv_path, index=False)

    src = {
        "source": "T", "file": str(csv_path), "date_cols": ["date"],
        "frequency": "daily", "coverage_logic": "from_row", "notes": "",
    }
    monkeypatch.setattr(dq, "SOURCES", [src])
    df = dq.compute_data_quality(reference_date=pd.Timestamp("2026-10-16"))
    row = df[df["source"] == "T"].iloc[-1]
    assert row["coverage"] == pytest.approx(0.9944)