"""REP-AUDIT-20261020-EU-DATE: european_coverage escribe date = sesion NYSE.

Antes: _ref_to_date devolvia run_date (Madrid). Runs en findes o
madrugada del martes escribian 'date' no bursatil (459 filas
detectadas en auditoria 2026-10-20).

Ahora: date = last_expected_market_date(reference_date).
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.european_coverage import _ref_to_date
from src.market_calendar import is_market_day


CASOS = [
    ("2026-10-19 23:30:00", "2026-10-19", "lunes noche Madrid (USA cerro) -> lunes"),
    ("2026-10-19 15:00:00", "2026-10-16", "lunes tarde Madrid (USA no cerro) -> viernes"),
    ("2026-10-20 02:50:00", "2026-10-19", "martes madrugada Madrid -> lunes"),
    ("2026-10-17 02:50:00", "2026-10-16", "sabado madrugada -> viernes"),
    ("2026-10-18 15:00:00", "2026-10-16", "domingo tarde -> viernes"),
    ("2026-11-26 22:00:00", "2026-11-25", "Thanksgiving jueves -> miercoles"),
]


@pytest.mark.parametrize("ref,expected,desc", CASOS)
def test_ref_to_date_es_sesion_bursatil(ref, expected, desc):
    result = _ref_to_date(pd.Timestamp(ref))
    assert result == pd.Timestamp(expected).date(), f"{desc}: {result} != {expected}"


def test_ref_to_date_siempre_bursatil():
    for ref, _, _ in CASOS:
        result = _ref_to_date(pd.Timestamp(ref))
        assert is_market_day(result), f"date={result} no bursatil (ref={ref})"


def test_ref_to_date_none_no_futura():
    """None -> last_expected_market_date(now), nunca futura respecto a sesion."""
    result = _ref_to_date(None)
    assert is_market_day(result)
    from src.market_calendar import last_expected_market_date
    assert result <= last_expected_market_date()