# -*- coding: utf-8 -*-
"""Tests de last_expected_lse_session (F-IAE-LSE-INTEGRATION, subciclo 1a).

Ventana real del radar: 4 slots UTC (23:17, 03:17, 07:17, 11:17).
Equivalen en Madrid (verano, CEST=UTC+2) a:
  01:17, 05:17, 09:17, 13:17
Todos dentro de la ventana "madrugada/mañana Madrid".

En esa ventana, last_expected_lse_session coincide con
last_expected_market_date para dias NYSE normales. En momentos
artificiales (tarde Madrid, antes de PUBLISH_HOUR=23) puede diferir;
esos casos se documentan aqui como esperados.

Limitacion conocida: is_trading_session("LSE", d) es lunes-viernes,
sin festivos UK. LSE expected == NYSE expected en dias normales.
"""
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.market_hours import last_expected_lse_session
from src.market_calendar import last_expected_market_date


MAD = ZoneInfo("Europe/Madrid")


# ---------------- Ventana real del radar ----------------

def test_slot_1_madrugada_martes():
    """Slot 1 (23:17 UTC) en Madrid -> 01:17 martes."""
    ref = datetime(2026, 9, 29, 1, 17, tzinfo=MAD)
    assert last_expected_lse_session(ref).isoformat() == "2026-09-28"


def test_slot_2_madrugada_martes():
    """Slot 2 (03:17 UTC) -> 05:17 martes."""
    ref = datetime(2026, 9, 29, 5, 17, tzinfo=MAD)
    assert last_expected_lse_session(ref).isoformat() == "2026-09-28"


def test_slot_3_manana_martes():
    """Slot 3 (07:17 UTC) -> 09:17 martes."""
    ref = datetime(2026, 9, 29, 9, 17, tzinfo=MAD)
    assert last_expected_lse_session(ref).isoformat() == "2026-09-28"


def test_slot_4_manana_martes():
    """Slot 4 (11:17 UTC) -> 13:17 martes."""
    ref = datetime(2026, 9, 29, 13, 17, tzinfo=MAD)
    assert last_expected_lse_session(ref).isoformat() == "2026-09-28"


def test_coincide_con_nyse_en_ventana_radar():
    """En la ventana del radar, LSE y NYSE coinciden (sin festivos UK)."""
    slots = [
        datetime(2026, 9, 29, 1, 17, tzinfo=MAD),
        datetime(2026, 9, 29, 5, 17, tzinfo=MAD),
        datetime(2026, 9, 29, 9, 17, tzinfo=MAD),
        datetime(2026, 9, 29, 13, 17, tzinfo=MAD),
    ]
    for ref in slots:
        assert last_expected_lse_session(ref) == last_expected_market_date(ref)


# ---------------- Fin de semana ----------------

def test_sabado_devuelve_viernes():
    ref = datetime(2026, 9, 26, 12, 0, tzinfo=MAD)
    assert last_expected_lse_session(ref).isoformat() == "2026-09-25"


def test_domingo_devuelve_viernes():
    ref = datetime(2026, 9, 27, 12, 0, tzinfo=MAD)
    assert last_expected_lse_session(ref).isoformat() == "2026-09-25"


def test_lunes_madrugada_devuelve_viernes():
    """Lunes 01:00 Madrid: LSE del lunes no ha abierto aun."""
    ref = datetime(2026, 9, 28, 1, 0, tzinfo=MAD)
    assert last_expected_lse_session(ref).isoformat() == "2026-09-25"


def test_lunes_tarde_devuelve_lunes():
    """Lunes 18:00 Madrid: LSE cerro a las 17:30."""
    ref = datetime(2026, 9, 28, 18, 0, tzinfo=MAD)
    assert last_expected_lse_session(ref).isoformat() == "2026-09-28"


# ---------------- Momentos artificiales (documentados) ----------------

def test_martes_tarde_difiere_de_nyse():
    """Martes 18:00 Madrid: LSE cerro, NYSE no.

    Este momento NO ocurre en produccion (cron del radar:
    01:17-13:17 Madrid). Se documenta el comportamiento correcto:
    LSE espera la sesion de hoy (ya cerrada); NYSE espera la de ayer
    (PUBLISH_HOUR=23 no ha pasado).
    """
    ref = datetime(2026, 9, 29, 18, 0, tzinfo=MAD)
    lse = last_expected_lse_session(ref)
    nyse = last_expected_market_date(ref)
    assert lse.isoformat() == "2026-09-29"
    assert nyse.isoformat() == "2026-09-28"
    assert lse != nyse


# ---------------- Precondiciones ----------------

def test_naive_datetime_falla():
    with pytest.raises(ValueError, match="timezone-aware"):
        last_expected_lse_session(datetime(2026, 9, 29, 12, 0))


def test_none_falla():
    with pytest.raises(ValueError, match="timezone-aware"):
        last_expected_lse_session(None)


# ---------------- Precio especifico: cierre LSE 16:30 London ----------------

def test_antes_cierre_lse_devuelve_dia_anterior():
    """Martes 16:00 Madrid: LSE aun abierto (cierra 17:30 Madrid)."""
    ref = datetime(2026, 9, 29, 16, 0, tzinfo=MAD)
    assert last_expected_lse_session(ref).isoformat() == "2026-09-28"


def test_despues_cierre_lse_devuelve_dia_actual():
    """Martes 18:00 Madrid: LSE ya cerro."""
    ref = datetime(2026, 9, 29, 18, 0, tzinfo=MAD)
    assert last_expected_lse_session(ref).isoformat() == "2026-09-29"
