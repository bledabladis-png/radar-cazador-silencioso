"""REP-AUDIT-20261018-DQ-DATE: data_quality.csv con date = sesion bursatil.

Antes: 'date' = fecha del run (Madrid). Generaba filas en sabados,
domingos y festivos NYSE. Ahora 'date' = last_expected_market_date(
reference_date), con lag PUBLISH_HOUR.

Contrato: ninguna fila de data_quality.csv puede tener 'date' no
bursatil, con independencia del momento del run.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators import data_quality as dq
from src.market_calendar import is_market_day


CASOS = [
    # (reference_date, expected_session, descripcion)
    ("2026-10-17 04:39:00", "2026-10-16", "sabado madrugada Madrid -> viernes"),
    ("2026-10-18 01:40:00", "2026-10-16", "domingo madrugada Madrid -> viernes"),
    ("2026-10-18 16:34:00", "2026-10-16", "domingo tarde Madrid -> viernes"),
    ("2026-10-19 10:00:00", "2026-10-16", "lunes manana Madrid -> viernes (USA no cerro)"),
    ("2026-10-19 23:30:00", "2026-10-19", "lunes noche Madrid -> lunes (USA cerro)"),
    ("2026-10-20 01:17:00", "2026-10-19", "martes madrugada Madrid -> lunes"),
    ("2026-10-20 23:30:00", "2026-10-20", "martes noche Madrid -> martes"),
    # Festivo NYSE: Thanksgiving 2026 = 26 nov (jueves)
    ("2026-11-26 23:30:00", "2026-11-25", "Thanksgiving jueves -> miercoles"),
    ("2026-11-27 10:00:00", "2026-11-25", "viernes post-Thanksgiving manana -> miercoles"),
]


@pytest.mark.parametrize("ref,expected_session,desc", CASOS)
def test_date_es_sesion_bursatil(ref, expected_session, desc):
    """Con reference_date date/time, 'date' del CSV = sesion bursatil."""
    df = dq.compute_data_quality(reference_date=pd.Timestamp(ref))
    fechas = df["date"].unique()
    assert len(fechas) == 1, f"{desc}: date no unico {fechas}"
    assert fechas[0] == expected_session, f"{desc}: {fechas[0]} != {expected_session}"


def test_date_siempre_bursatil_en_todos_los_casos():
    """Invariante: date es dia bursatil en todos los casos."""
    for ref, _, _ in CASOS:
        df = dq.compute_data_quality(reference_date=pd.Timestamp(ref))
        for fecha_str in df["date"].unique():
            d = pd.Timestamp(fecha_str).date()
            assert is_market_day(d), f"date={fecha_str} no bursatil (ref={ref})"


def test_age_calculado_contra_sesion_no_contra_run():
    """age = sesion - last_date, no run_date - last_date."""
    ref = pd.Timestamp("2026-10-18 01:40:00")
    df = dq.compute_data_quality(reference_date=ref)
    # Fila FINRA: last_date suele ser anterior al viernes.
    finra = df[df["source"] == "FINRA Dark Pool"]
    assert not finra.empty
    row = finra.iloc[0]
    if pd.notna(row["last_date"]) and pd.notna(row["age_calendar_days"]):
        expected = (pd.Timestamp("2026-10-16") - pd.Timestamp(row["last_date"])).days
        assert int(row["age_calendar_days"]) == expected