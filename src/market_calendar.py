# -*- coding: utf-8 -*-
"""Calendario de dias de mercado USA (NYSE).

Usado por los loaders para detectar caches con datos obsoletos.
Festivos NYSE 2026-2027 (cierre total).
"""
from datetime import datetime, timedelta

NYSE_HOLIDAYS = {
    # 2026
    "2026-01-01",  # New Year
    "2026-01-19",  # MLK
    "2026-02-16",  # Presidents
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # July 4 observed (Sat)
    "2026-09-07",  # Labor
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
    # 2027
    "2027-01-01",  # New Year
    "2027-01-18",  # MLK
    "2027-02-15",  # Presidents
    "2027-03-26",  # Good Friday
    "2027-05-31",  # Memorial
    "2027-06-18",  # Juneteenth observed (Sat)
    "2027-07-05",  # July 4 observed (Sun)
    "2027-09-06",  # Labor
    "2027-11-25",  # Thanksgiving
    "2027-12-24",  # Christmas observed (Sat)
}

# Yahoo publica el cierre EOD con lag respecto al cierre de NYSE
# (16:00 ET = 22:00 CEST). Asumimos cierre disponible a las PUBLISH_HOUR
# hora local (Madrid). Antes de esa hora, el ultimo dato esperado es
# del dia de mercado anterior.
PUBLISH_HOUR = 23


def is_market_day(d):
    """True si `d` (date) es dia de mercado NYSE."""
    if d.weekday() >= 5:
        return False
    if d.strftime("%Y-%m-%d") in NYSE_HOLIDAYS:
        return False
    return True


def previous_market_day(d):
    """Retrocede hasta el dia de mercado anterior a `d` (exclusive)."""
    d = d - timedelta(days=1)
    while not is_market_day(d):
        d = d - timedelta(days=1)
    return d


def last_expected_market_date(now=None):
    """Ultima fecha de mercado esperada a `now`.

    Acepta `datetime` (recomendado, con hora para respetar PUBLISH_HOUR)
    o `date` (asume hora 0 -> SIEMPRE aplica el lag de publicacion).

    Nota B5 (2026-09-12): para un `date` sin hora, la funcion es
    conservadora y retrocede un dia natural antes de buscar el ultimo
    dia de mercado. Ejemplo:
        last_expected_market_date(date(2026, 9, 14)) == date(2026, 9, 11)
        (lunes -> domingo -> sabado -> viernes; NO devuelve el propio lunes)

    Considera el lag de publicacion de Yahoo: antes de PUBLISH_HOUR,
    el cierre del dia anterior todavia no esta disponible.
    """
    if now is None:
        now = datetime.now()
    # B5 fix (2026-09-12): normalizar entrada datetime/date/Timestamp.
    # datetime es subclase de date; el orden del isinstance importa.
    if isinstance(now, datetime):
        d = now.date()
        hour = now.hour
    else:
        # datetime.date (o compatible)
        d = now
        hour = 0
    # Si es antes de PUBLISH_HOUR, retroceder un dia
    if hour < PUBLISH_HOUR:
        d = d - timedelta(days=1)
    # Retroceder hasta dia de mercado
    while not is_market_day(d):
        d = d - timedelta(days=1)
    return d