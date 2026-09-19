"""OpenFIGI API client productivo.

Endpoint: https://api.openfigi.com/v3/mapping

Soporta:
  - ID_CUSIP     (recomendado para el universo 13F)
  - TICKER       (recomendado para el radar)
  - ID_ISIN
  - ID_EXCH_SYMBOL

Contrato:
  - Sin estado global. Funcion pura sobre inputs.
  - Batching configurable.
  - Reintentos SOLO en 429, 500, 503. Fallo inmediato en 4xx.
  - Sin datetime.now(). La fecha la aporta el caller.

Devolucion:
  {
    "<id_value>": {
      "ok": bool,
      "data": list[dict] | None,      # figi, shareClassFIGI, ticker, name, ...
      "error": str | None,
    },
    ...
  }
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

URL = "https://api.openfigi.com/v3/mapping"
UA = "Macro_Sectorial-research/1.0"
VALID_ID_TYPES = (
    "ID_CUSIP",
    "ID_ISIN",
    "ID_EXCH_SYMBOL",
    "TICKER",
)
RETRYABLE_HTTP = (429, 500, 503)


def _build_headers(api_key: str | None) -> dict:
    h = {
        "Content-Type": "application/json",
        "User-Agent": UA,
        "Accept": "application/json",
    }
    if api_key:
        h["X-OPENFIGI-APIKEY"] = api_key
    return h


def _batch_size(api_key: str | None) -> int:
    return 100 if api_key else 5


def _sleep_seconds(api_key: str | None) -> float:
    return 1.0 if api_key else 2.5


def _post(payload, headers, *, max_retry: int) -> list:
    body = json.dumps(payload).encode("utf-8")
    attempt = 0
    while True:
        try:
            req = urllib.request.Request(URL, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in RETRYABLE_HTTP and attempt < max_retry:
                attempt += 1
                time.sleep(5 * attempt)
                continue
            msg = e.read().decode("utf-8")[:200]
            raise RuntimeError(f"HTTP {e.code}: {msg}") from e


def map_identifiers(
    id_type: str,
    values: list[str],
    *,
    exch_code: str | None = None,
    api_key: str | None = None,
    max_retry: int = 2,
) -> dict:
    """Mapea una lista de identificadores a traves de OpenFIGI /v3/mapping.

    Args:
        id_type: uno de VALID_ID_TYPES.
        values: lista de identificadores del tipo indicado.
        exch_code: opcional, ej. "US".
        api_key: si None, usa OPENFIGI_API_KEY del entorno.
        max_retry: reintentos en 429/500/503.

    Returns:
        dict por value. Ver docstring del modulo.
    """
    if id_type not in VALID_ID_TYPES:
        raise ValueError(f"id_type invalido: {id_type}. Validos: {VALID_ID_TYPES}")

    clean = [str(v).strip() for v in values if v is not None and str(v).strip()]
    if not clean:
        return {}

    key = api_key if api_key is not None else os.environ.get("OPENFIGI_API_KEY", "").strip()
    key = key or None
    headers = _build_headers(key)
    batch = _batch_size(key)
    sleep_s = _sleep_seconds(key)

    out: dict = {}
    for i in range(0, len(clean), batch):
        chunk = clean[i:i + batch]
        payload = []
        for v in chunk:
            job = {"idType": id_type, "idValue": v}
            if exch_code:
                job["exchCode"] = exch_code
            payload.append(job)

        try:
            parsed = _post(payload, headers, max_retry=max_retry)
        except Exception as e:
            for v in chunk:
                out[v] = {"ok": False, "data": None, "error": f"{type(e).__name__}: {e}"}
            if i + batch < len(clean):
                time.sleep(sleep_s)
            continue

        for v, resp in zip(chunk, parsed):
            if isinstance(resp, dict) and "data" in resp:
                out[v] = {"ok": True, "data": resp["data"], "error": None}
            elif isinstance(resp, dict) and "error" in resp:
                out[v] = {"ok": False, "data": None, "error": str(resp["error"])}
            elif isinstance(resp, dict) and "warning" in resp:
                out[v] = {"ok": False, "data": None, "error": str(resp["warning"])}
            else:
                out[v] = {"ok": False, "data": None, "error": f"unknown shape: {resp}"}

        if i + batch < len(clean):
            time.sleep(sleep_s)

    return out


def extract_stable_identity(hit: dict | None) -> dict | None:
    """Normaliza un hit de OpenFIGI a la identidad estable que guarda el catalogo.

    Devuelve dict con:
      figi, share_class_figi, ticker, name,
      security_type, market_sector, exch_code
    o None si no hay hit valido.
    """
    if not hit or not hit.get("ok") or not hit.get("data"):
        return None
    rows = hit["data"]
    if not isinstance(rows, list) or not rows:
        return None
    # Si hay multiples rows, tomar la primera con exchCode US.
    chosen = None
    for r in rows:
        if r.get("exchCode") == "US":
            chosen = r
            break
    if chosen is None:
        chosen = rows[0]
    return {
        "figi": chosen.get("figi"),
        "share_class_figi": chosen.get("shareClassFIGI"),
        "composite_figi": chosen.get("compositeFIGI"),
        "ticker": chosen.get("ticker"),
        "name": chosen.get("name"),
        "security_type": chosen.get("securityType"),
        "market_sector": chosen.get("marketSector"),
        "exch_code": chosen.get("exchCode"),
    }
