# tests/test_fu001_sin_ffill_global.py
"""Tests FU-001 (2026-09-15): el loader NO debe rellenar globalmente
los NaN preservados por B1 en el merge consolidado.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.stock_data_loader import _fill_holes_respecting_sessions


def test_no_ffill_global_en_loader():
    """Regresion estructural: 'data.ffill(limit=3)' NO debe existir
    en src/stock_data_loader.py. Si alguien lo reintroduce, este test falla."""
    src = Path("src/stock_data_loader.py").read_text(encoding="utf-8-sig")
    assert "data.ffill(limit=3)" not in src, (
        "FU-001 regresion: 'data.ffill(limit=3)' vuelve a estar presente. "
        "El ffill global deshace la proteccion B1."
    )


def test_b1_rellena_nan_en_fin_de_semana():
    """B1 debe RELLENAR NaN cuando la fecha NO es sesion NYSE (fin de semana)."""
    # Sabado 2026-09-12 y domingo 2026-09-13 son fin de semana.
    # Viernes 2026-09-11 es dia de bolsa.
    dates = pd.to_datetime(["2026-09-11", "2026-09-12", "2026-09-13"])
    df = pd.DataFrame({
        ("Close", "TEST"): [100.0, np.nan, np.nan],
    }, index=dates)
    ref = datetime(2026, 9, 14, 10, 0)
    filled, diag = _fill_holes_respecting_sessions(df, ref)

    # Sabado y domingo NO son sesion NYSE -> B1 los rellena
    assert not pd.isna(filled.loc["2026-09-12", ("Close", "TEST")])
    assert not pd.isna(filled.loc["2026-09-13", ("Close", "TEST")])
    assert filled.loc["2026-09-12", ("Close", "TEST")] == 100.0
    assert filled.loc["2026-09-13", ("Close", "TEST")] == 100.0
    # No debe haber NaN preservados (no habia ningun NaN en dia bursatil)
    assert diag["n_nan_preserved"] == 0


def test_b1_preserva_nan_en_expected_session():
    """B1 debe preservar NaN cuando la fecha SI es la sesion esperada."""
    dates = pd.to_datetime(["2026-09-10", "2026-09-11"])
    df = pd.DataFrame({
        ("Close", "TEST"): [100.0, np.nan],
    }, index=dates)
    # reference_date = viernes 11 post-cierre -> expected_session = viernes 11
    ref = datetime(2026, 9, 11, 23, 30)
    filled, diag = _fill_holes_respecting_sessions(df, ref)

    # El NaN del viernes ES sesion esperada -> B1 lo preserva
    assert pd.isna(filled.loc["2026-09-11", ("Close", "TEST")])
    assert diag["n_nan_preserved"] >= 1
