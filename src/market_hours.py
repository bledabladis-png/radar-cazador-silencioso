# -*- coding: utf-8 -*-
"""Modelo temporal minimo para elegibilidad EOD (FU-018).

Determina si la sesion de una fecha ya cerro, para decidir si una
observacion del provider es un cierre EOD valido o una vela en curso.

NO es un calendario oficial completo. Es el minimo necesario para
filtrar Yahoo USA/UK y Xetra cuando el mercado esta abierto.
"""
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.market_calendar import is_market_day


class UnknownMarketError(Exception):
    """Raised when a market is not recognized."""
    pass


_REGULAR_CLOSE_PATH = Path("config/market_close_regular.csv")
_EXCEPTIONS_PATH = Path("config/market_close_exceptions.csv")

_KNOWN_MARKETS = ("US_EQUITY", "LSE", "XETRA", "BME", "EURONEXT")

_regular_cache = None
_exceptions_cache = None


def _load_regular_close():
    global _regular_cache
    if _regular_cache is not None:
        return _regular_cache
    if not _REGULAR_CLOSE_PATH.exists():
        raise FileNotFoundError(f"Falta {_REGULAR_CLOSE_PATH}")
    df = pd.read_csv(_REGULAR_CLOSE_PATH)
    _regular_cache = {}
    for _, row in df.iterrows():
        _regular_cache[row["market"]] = {
            "tz": ZoneInfo(row["timezone"]),
            "regular_close": dtime.fromisoformat(row["regular_close"]),
        }
    return _regular_cache


def _load_exceptions():
    global _exceptions_cache
    if _exceptions_cache is not None:
        return _exceptions_cache
    if not _EXCEPTIONS_PATH.exists():
        _exceptions_cache = {}
        return _exceptions_cache
    df = pd.read_csv(_EXCEPTIONS_PATH)
    _exceptions_cache = {}
    if df.empty:
        return _exceptions_cache
    for _, row in df.iterrows():
        _exceptions_cache[(row["market"], str(row["date"]))] = dtime.fromisoformat(row["close"])
    return _exceptions_cache


def _normalize_date(d):
    """Acepta date/datetime/Timestamp/str y devuelve date."""
    if hasattr(d, "date") and callable(d.date) and not isinstance(d, str):
        return d.date()
    if isinstance(d, str):
        return pd.Timestamp(d).date()
    return d


def last_expected_lse_session(reference_date):
    """Ultima sesion LSE cerrada antes o en reference_date.

    Itera hacia atras desde reference_date.date(). Devuelve la primera
    fecha d que cumple:
      - is_trading_session("LSE", d) == True
      - is_session_closed("LSE", d, reference_date) == True

    Contraste con src.market_calendar.last_expected_market_date():
      - Aquel usa calendario NYSE + PUBLISH_HOUR (Madrid, 23:00).
      - Este usa horario de cierre LSE (16:30 London) y calendario LSE
        (lunes-viernes).

    En dias normales coinciden. En festivos NYSE (LSE abierto) pueden
    diferir. En festivos UK (LSE cerrado, NYSE abierto) NO estan
    contemplados: is_trading_session("LSE", d) es lunes-viernes sin
    festivos UK. Limitacion documentada.

    Precondicion: reference_date debe ser timezone-aware.
    """
    if reference_date is None or reference_date.tzinfo is None:
        raise ValueError(
            "reference_date must be timezone-aware. "
            "Pass datetime.now(ZoneInfo('Europe/Madrid')) or equivalent."
        )
    d = reference_date.date()
    # Limite de seguridad: 30 dias. Sin esto, un reference_date
    # patologico (todo festivo) provocaria un bucle infinito.
    for _ in range(30):
        if is_trading_session("LSE", d) and \
           is_session_closed("LSE", d, reference_date):
            return d
        d = d - timedelta(days=1)
    raise RuntimeError(
        "last_expected_lse_session: no se encontro sesion LSE cerrada "
        "en los ultimos 30 dias desde {0}".format(reference_date)
    )


def is_trading_session(market: str, session_date) -> bool:
    """True si session_date es dia de negociacion en market.

    - US_EQUITY: calendario oficial NYSE (is_market_day).
    - LSE/XETRA/BME/EURONEXT: PROVISIONAL, lunes-viernes.
      NO es calendario oficial. Festivos locales no contemplados.
    - Otro: UnknownMarketError.
    """
    if market not in _KNOWN_MARKETS:
        raise UnknownMarketError(f"Market desconocido: {market!r}")
    d = _normalize_date(session_date)
    if market == "US_EQUITY":
        return is_market_day(d)
    # PROVISIONAL: lunes-viernes. Festivos no contemplados.
    return d.weekday() < 5


def get_session_close(market: str, session_date) -> datetime:
    """Hora de cierre efectiva de session_date en market (tz-aware).

    - Si session_date no es dia de negociacion -> ValueError.
    - Aplica excepciones por fecha si existen.
    """
    if market not in _KNOWN_MARKETS:
        raise UnknownMarketError(f"Market desconocido: {market!r}")
    d = _normalize_date(session_date)
    if not is_trading_session(market, d):
        raise ValueError(
            f"{d} no es dia de negociacion en {market}. "
            "No existe cierre EOD valido para esa fecha."
        )
    regular = _load_regular_close()
    if market not in regular:
        raise UnknownMarketError(f"Market {market!r} no esta en market_close_regular.csv")
    tz = regular[market]["tz"]
    close_time = regular[market]["regular_close"]

    exceptions = _load_exceptions()
    key = (market, str(d))
    if key in exceptions:
        close_time = exceptions[key]

    return datetime.combine(d, close_time, tzinfo=tz)


def is_session_closed(market: str, session_date, reference_date) -> bool:
    """True si la sesion de session_date en market ya cerro en reference_date.

    Precondiciones:
    - reference_date debe ser timezone-aware. Si no -> ValueError.
    - market desconocido -> UnknownMarketError.
    - session_date debe ser dia de negociacion. Si no -> ValueError.
    """
    if reference_date is None or reference_date.tzinfo is None:
        raise ValueError(
            "reference_date must be timezone-aware. "
            "Pass datetime.now(timezone.utc) or equivalent."
        )
    if market not in _KNOWN_MARKETS:
        raise UnknownMarketError(f"Market desconocido: {market!r}")
    d = _normalize_date(session_date)
    if not is_trading_session(market, d):
        raise ValueError(
            f"{d} no es dia de negociacion en {market}. "
            "is_session_closed no aplica."
        )
    ref_d = reference_date.date()
    if d < ref_d:
        return True
    if d > ref_d:
        print(f"  [WARN] session_date={d} > reference_date.date()={ref_d}")
        return False
    close_dt = get_session_close(market, d)
    return reference_date >= close_dt
