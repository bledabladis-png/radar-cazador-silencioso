# -*- coding: utf-8 -*-
"""Loader de los JSON producidos por el scraper LSE (lse-close-scraper).

Contrato JSON congelado 2026-09-25 (dictamen auditor externo):

  Top-level:  request / data / status / metadata
  data:       lista de {_DATE_END, OPEN_PRC, HIGH_1, LOW_1, CLOSE_PRC}
  Unidad:     GBX (peniques). NO se convierte.
  Fechas:     _DATE_END string ISO YYYY-MM-DD.
  Precios:    string float (GBX).
  status:     "OK" cuando hay datos; substatus "BackendError" si el
              RIC no existe (data=[]).

Reglas de diseno:
  - Sin datetime.now(). La fecha viene por parametro.
  - Sin conversion de unidad. stock_prices.parquet ya esta en GBX
    para los .L (confirmado empiricamente 2026-09-25).
  - Sin volumen. El scraper no lo trae; en la integracion vendra
    de Yahoo.
  - Devuelve por observacion: solo la fila de expected_session.
  - Tolerante a fallos: los tickers sin datos validos se omiten.

Uso:
    datos_dir = Path("data/external/lse_close/datos")
    session = date(2026, 9, 25)
    tickers = ["BA.L", "AZN.L"]
    rows = load_lse_close_for_session(datos_dir, session, tickers)
"""
from __future__ import annotations

import json
from pathlib import Path

from src.instrument_registry import INSTRUMENTS


def _ric_for_ticker(ticker):
    """Devuelve el RIC Refinitiv del ticker o None si no aplica."""
    inst = INSTRUMENTS.get(ticker)
    if not isinstance(inst, dict):
        return None
    ric = inst.get("refinitiv")
    if isinstance(ric, str) and ric:
        return ric
    return None


def _read_json(datos_dir, ric):
    """Lee el JSON del RIC. Devuelve dict o None si falla."""
    filename = ric.replace(".", "_") + ".json"
    path = datos_dir / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _to_float(value):
    """Convierte string a float. Devuelve None si falla."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return None


def _extract_session_row(json_data, expected_session):
    """Busca la fila de expected_session en el JSON.

    Devuelve dict con {date, open, high, low, close} o None si no hay
    observacion valida para esa fecha exacta.

    Reglas:
      - expected_session es date ISO. NO se usa max(_DATE_END) como
        sustituto. Debe existir esa fecha exacta.
      - Close debe ser numerico no-NaN. Los OHLC ausentes se devuelven
        como None, pero Close es obligatorio.
    """
    if not isinstance(json_data, dict):
        return None
    data = json_data.get("data")
    if not isinstance(data, list):
        return None

    target = expected_session.isoformat() if hasattr(expected_session, "isoformat") else str(expected_session)

    for row in data:
        if not isinstance(row, dict):
            continue
        if row.get("_DATE_END") != target:
            continue
        close = _to_float(row.get("CLOSE_PRC"))
        if close is None:
            return None
        return {
            "date": target,
            "open": _to_float(row.get("OPEN_PRC")),
            "high": _to_float(row.get("HIGH_1")),
            "low": _to_float(row.get("LOW_1")),
            "close": close,
        }
    return None


def load_lse_close_for_session(datos_dir, expected_session, tickers):
    """Carga la observacion de expected_session para los tickers dados.

    Parametros:
        datos_dir: Path al directorio con los JSON del scraper.
        expected_session: date de la sesion LSE esperada (FU-018).
        tickers: lista de tickers canonicos del radar (p.ej. ["BA.L"]).

    Devuelve:
        dict[ticker, {date, open, high, low, close}]
        Solo incluye tickers con observacion valida en esa fecha.
        Los tickers sin RIC, sin fichero o sin la fecha esperada
        se omiten (no se genera entrada).
    """
    result = {}
    if not tickers:
        return result

    datos_dir = Path(datos_dir)

    for ticker in tickers:
        ric = _ric_for_ticker(ticker)
        if ric is None:
            continue
        json_data = _read_json(datos_dir, ric)
        if json_data is None:
            continue
        row = _extract_session_row(json_data, expected_session)
        if row is None:
            continue
        result[ticker] = row

    return result
