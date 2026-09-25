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


def build_lse_close_override(loaded_dict):
    """Construye un DataFrame con solo ('Close', ticker) para override.

    Entrada: dict[ticker, {date, open, high, low, close}] tal como
    devuelve load_lse_close_for_session. Todas las entradas deben
    tener la MISMA date (una sola sesion por llamada).

    Salida: DataFrame con MultiIndex de una fila:
        columns = MultiIndex.from_product([['Close'], tickers])
        index = DatetimeIndex([pd.Timestamp(date)])

    El DataFrame resultante se combinara con el de Yahoo para
    sobrescribir Close, preservando Open/High/Low/Volume.

    Raises:
        ValueError si las fechas no son homogeneas.
        ValueError si el dict esta vacio.

    Sin conversion de unidad (GBX). Sin datetime.now().
    """
    import pandas as pd

    if not loaded_dict:
        raise ValueError("build_lse_close_override: dict vacio")

    dates = {row["date"] for row in loaded_dict.values()}
    if len(dates) != 1:
        raise ValueError(
            "build_lse_close_override: fechas no homogeneas: {0}".format(
                sorted(dates))
        )

    fecha = next(iter(dates))
    tickers = sorted(loaded_dict.keys())
    close_values = [loaded_dict[t]["close"] for t in tickers]

    df = pd.DataFrame(
        [close_values],
        index=pd.DatetimeIndex([pd.Timestamp(fecha)]),
        columns=pd.MultiIndex.from_product([["Close"], tickers]),
    )
    return df


def write_lse_provenance(
    path,
    *,
    target_session,
    lse_expected_session,
    source_repo,
    source_ref,
    source_commit,
    tickers_from_scraper,
    tickers_from_yahoo,
    run_id,
):
    """Escribe provenance persistente del uso del scraper LSE.

    Dictamen auditor externo 2026-09-25:
      - Persistente y versionado junto al parquet.
      - source_commit obligatorio si el scraper se uso (tickers_from_scraper
        no vacio). En local, sin scraper, puede ser None.
      - Si el scraper se usa y source_commit esta vacio -> ValueError.

    Escritura atomica: tmp + os.replace. Sin datetime.now().

    Args:
        path: destino (Path o str). Por convencion:
              'data/lse_close_provenance.json'.
        target_session: fecha global objetivo del pipeline (date|str).
        lse_expected_session: sesion LSE especifica (date|str).
        source_repo: 'bledabladis-png/lse-close-scraper'.
        source_ref: 'main' u otra ref.
        source_commit: SHA del commit del scraper consumido. None o ''
                       solo si tickers_from_scraper esta vacio.
        tickers_from_scraper: lista de tickers (radar) que vinieron del scraper.
        tickers_from_yahoo: lista de tickers .L que cayeron a Yahoo.
        run_id: 'YYYYMMDD_HHMMSS'.

    Returns:
        dict con el payload escrito.
    """
    import json
    import os as _os
    from pathlib import Path as _Path

    scraper_used = bool(tickers_from_scraper)
    if scraper_used and not source_commit:
        raise ValueError(
            "write_lse_provenance: source_commit obligatorio cuando el "
            "scraper se ha usado (dictamen D4). "
            "tickers_from_scraper={0}, source_commit={1!r}".format(
                len(tickers_from_scraper), source_commit)
        )

    def _iso(d):
        if d is None:
            return None
        if hasattr(d, "isoformat"):
            return d.isoformat()
        return str(d)

    payload = {
        "target_session": _iso(target_session),
        "lse_expected_session": _iso(lse_expected_session),
        "source_repo": source_repo,
        "source_ref": source_ref,
        "source_commit": source_commit if source_commit else None,
        "tickers_from_scraper": sorted(tickers_from_scraper or []),
        "tickers_from_yahoo": sorted(tickers_from_yahoo or []),
        "run_id": run_id,
        "scraper_used": scraper_used,
    }

    out_path = _Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(out_path) + ".tmp." + (run_id or "norunid")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
    _os.replace(tmp_path, str(out_path))

    return payload
