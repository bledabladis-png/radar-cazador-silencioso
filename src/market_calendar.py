# -*- coding: utf-8 -*-
"""Calendario de dias de mercado USA (NYSE).

Usado por los loaders para detectar caches con datos obsoletos.

Festivos NYSE calculados algoritmicamente para cualquier ano
(reglas federales + NYSE). Sin hardcoding por ano.
"""
from datetime import date, datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd

_MADRID = ZoneInfo("Europe/Madrid")

# Yahoo publica el cierre EOD con lag respecto al cierre de NYSE
# (16:00 ET = 22:00 CEST). Asumimos cierre disponible a las PUBLISH_HOUR
# hora Madrid. Antes de esa hora, el ultimo dato esperado es
# del dia de mercado anterior.
PUBLISH_HOUR = 23


def _easter_sunday(year: int) -> date:
    """Domingo de Pascua (Computus, rito gregoriano)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-esimo `weekday` (0=lunes) del mes (n >= 1)."""
    first = date(year, month, 1)
    delta = (weekday - first.weekday()) % 7
    return first + timedelta(days=delta + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Ultimo `weekday` (0=lunes) del mes."""
    if month == 12:
        last = date(year, 12, 31)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    delta = (last.weekday() - weekday) % 7
    return last - timedelta(days=delta)


def _observed(d: date) -> date:
    """Regla observed NYSE: sabado -> viernes; domingo -> lunes."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def _nyse_holidays(year: int) -> frozenset:
    """Festivos NYSE observados en asociacion con `year`.

    Nota: el observed de una fecha puede caer en un ano adyacente.
    Ejemplo: New Year 2028 (1-ene sabado) se observa el viernes 2027-12-31.
    Por eso `is_market_day` consulta tambien los anos adyacentes.
    """
    hol = set()
    hol.add(_observed(date(year, 1, 1)))               # New Year
    hol.add(_nth_weekday(year, 1, 0, 3))               # MLK (3er lunes enero)
    hol.add(_nth_weekday(year, 2, 0, 3))               # Presidents (3er lunes febrero)
    hol.add(_easter_sunday(year) - timedelta(days=2))  # Good Friday
    hol.add(_last_weekday(year, 5, 0))                 # Memorial (ultimo lunes mayo)
    if year >= 2022:
        hol.add(_observed(date(year, 6, 19)))          # Juneteenth
    hol.add(_observed(date(year, 7, 4)))               # Independence
    hol.add(_nth_weekday(year, 9, 0, 1))               # Labor (1er lunes septiembre)
    hol.add(_nth_weekday(year, 11, 3, 4))              # Thanksgiving (4o jueves noviembre)
    hol.add(_observed(date(year, 12, 25)))             # Christmas
    return frozenset(hol)


def _is_nyse_holiday(d: date) -> bool:
    """True si `d` es festivo NYSE observado (considera anos adyacentes).

    Necesario por el caso New Year sabado: el observed cae en el ano anterior.
    """
    if d in _nyse_holidays(d.year):
        return True
    if d in _nyse_holidays(d.year + 1):
        return True
    if d.year > 1 and d in _nyse_holidays(d.year - 1):
        return True
    return False


def is_market_day(d):
    """True si `d` es dia de mercado NYSE.

    Acepta date, datetime, pandas.Timestamp o string ISO. Normaliza a date.
    """
    if isinstance(d, str):
        d = pd.Timestamp(d).date()
    elif isinstance(d, datetime):
        d = d.date()
    if d.weekday() >= 5:
        return False
    if _is_nyse_holiday(d):
        return False
    return True


def previous_market_day(d):
    """Retrocede hasta el dia de mercado anterior a `d` (exclusive)."""
    d = d - timedelta(days=1)
    limit = 366
    for _ in range(limit):
        if is_market_day(d):
            return d
        d = d - timedelta(days=1)
    raise RuntimeError(
        f"previous_market_day: sin dia de mercado en {limit} dias"
    )


def last_expected_market_date(now=None):
    """Ultima fecha de mercado esperada a `now` (hora Europe/Madrid).

    - now=None -> datetime.now(Europe/Madrid).
    - now datetime naive -> se interpreta como Europe/Madrid.
    - now datetime tz-aware -> se convierte a Europe/Madrid.
    - now date -> hora 0 -> siempre aplica el lag de PUBLISH_HOUR.

    Considera el lag de publicacion de Yahoo: antes de PUBLISH_HOUR
    (Madrid), el cierre del dia anterior todavia no esta disponible.
    """
    if now is None:
        now = datetime.now(_MADRID)
    if isinstance(now, datetime):
        if now.tzinfo is None:
            now = now.replace(tzinfo=_MADRID)
        else:
            now = now.astimezone(_MADRID)
        d = now.date()
        hour = now.hour
    else:
        d = now
        hour = 0
    if hour < PUBLISH_HOUR:
        d = d - timedelta(days=1)
    limit = 366
    for _ in range(limit):
        if is_market_day(d):
            return d
        d = d - timedelta(days=1)
    raise RuntimeError(
        f"last_expected_market_date: sin dia de mercado en {limit} dias"
    )


def _last_market_session(d):
    """FU-007/FU-007-b (2026-09-13): retrocede al ultimo dia bursatil <= d.

    Acepta datetime/Timestamp/date. No aplica lag de PUBLISH_HOUR
    (a diferencia de last_expected_market_date). Uso: readers que
    reciben una fecha que puede caer en fin de semana (FRED, Yahoo,
    parquet, cualquier CSV con fechas).
    """
    ts = pd.Timestamp(d)
    d_only = ts.date()
    limit = 366
    for _ in range(limit):
        if is_market_day(d_only):
            return pd.Timestamp(d_only)
        d_only = d_only - timedelta(days=1)
    raise RuntimeError(
        f"_last_market_session: sin dia de mercado en {limit} dias"
    )
