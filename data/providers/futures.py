"""Provider OilPriceAPI para futuros y spot (FU-021-3C-bis).

Universo:
  Futuros:  BZ=F (Brent front-month), CL=F (WTI front-month)
  Spot:     GC=F (Gold), HG=F (Copper), NG=F (Natural Gas)

Fuente: OilPriceAPI (https://api.oilpriceapi.com/v1)
  - /v1/futures/ice-brent
  - /v1/futures/ice-wti
  - /v1/prices/latest?by_code=GOLD_USD,COPPER_USD,NATURAL_GAS_USD

API key: env var OIL_PRICE_API (GH Actions secret).
Fallback local: D:/Descarga-Futuros/OilPriceApi/config/oilpriceapi-key.txt.

Presupuesto: 3 requests/dia. Plan free = 200/mes. Margen amplio.

Salida: dos parquets en data/commodities_*.parquet
  - commodities_futures.parquet   (BZ=F, CL=F)
  - commodities_spot.parquet      (GC=F, HG=F, NG=F)
Cada uno con manifest FU-002 (write_artifact_with_manifest).

Reglas:
  - Solo front-month de futuros (close = proxy de settlement oficial).
  - Sin fallback a Yahoo (Opcion A estricta del prompt Seccion 11.1).
  - Reintentos: 1 en timeout, 0 en 401/429.
  - Acumulacion historica: merge con parquet previo, dedupe por fecha.

Refs: FU-021-3C-bis, PROMPT_MAESTRO Seccion 11.8 (FU-002).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .base import MarketDataProvider


API_BASE = "https://api.oilpriceapi.com/v1"
DEFAULT_TIMEOUT = 30
LOCAL_KEY_FALLBACK = Path(
    "D:/Descarga-Futuros/OilPriceApi/config/oilpriceapi-key.txt"
)

FUTURES_MAP = {
    "BZ=F": {"endpoint": "/futures/ice-brent", "exchange": "ICE"},
    "CL=F": {"endpoint": "/futures/ice-wti", "exchange": "NYMEX"},
}

SPOT_MAP = {
    "GC=F": {"code": "GOLD_USD"},
    "HG=F": {"code": "COPPER_USD"},
    "NG=F": {"code": "NATURAL_GAS_USD"},
}

FIELDS = ["Open", "High", "Low", "Close", "Volume"]


class FuturesProvider(MarketDataProvider):
    name = "OilPriceAPI"

    def __init__(self):
        self.session = requests.Session()
        self._api_key = None

    # --- API key resolution --------------------------------------

    def _get_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        key = os.environ.get("OIL_PRICE_API", "").strip()
        if not key and LOCAL_KEY_FALLBACK.exists():
            key = LOCAL_KEY_FALLBACK.read_text(encoding="utf-8-sig").strip()
        if not key:
            raise RuntimeError(
                "OIL_PRICE_API no encontrado. Setear env var o crear "
                + str(LOCAL_KEY_FALLBACK)
            )
        self._api_key = key
        return key

    def _headers(self) -> dict:
        return {"Authorization": "Token " + self._get_api_key()}

    # --- HTTP con politica de reintentos -------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = API_BASE + path
        attempts = 0
        while True:
            attempts += 1
            try:
                r = self.session.get(
                    url, headers=self._headers(),
                    params=params, timeout=DEFAULT_TIMEOUT,
                )
            except requests.exceptions.Timeout:
                if attempts >= 2:
                    raise
                continue

            if r.status_code == 401:
                raise RuntimeError(
                    "OilPriceAPI 401 Unauthorized. Verificar API key."
                )
            if r.status_code == 429:
                raise RuntimeError(
                    "OilPriceAPI 429 Too Many Requests. "
                    "Limite mensual alcanzado. No reintentar."
                )
            if r.status_code >= 400:
                raise RuntimeError(
                    "OilPriceAPI " + str(r.status_code) + ": "
                    + r.text[:200]
                )
            return r.json()

    # --- Interfaz MarketDataProvider -----------------------------

    def get_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        try:
            self._get("/prices/latest", params={"by_code": "GOLD_USD"})
            return True
        except Exception as e:
            print("FuturesProvider.is_available: " + str(e))
            return False

    def get_prices(self, tickers, start=None, end=None, period="10y"):
        raise NotImplementedError(
            "FuturesProvider no usa get_prices. Usar fetch_commodities()."
        )

    def get_treasury_yields(self, maturities=None):
        raise NotImplementedError("FuturesProvider no provee yields.")

    def get_fed_data(self, series=None):
        raise NotImplementedError("FuturesProvider no provee datos Fed.")

    # --- Fetch ---------------------------------------------------

    def _fetch_futures_front_month(self, ticker: str) -> Optional[dict]:
        cfg = FUTURES_MAP[ticker]
        data = self._get(cfg["endpoint"])
        fm = data.get("front_month") or {}
        settlement_date = data.get("settlement_date")
        if not settlement_date or not fm:
            return None
        return {
            "date": settlement_date,
            "ticker": ticker,
            "Open": _to_float(fm.get("open")),
            "High": _to_float(fm.get("high")),
            "Low": _to_float(fm.get("low")),
            "Close": _to_float(fm.get("close")),
            "Volume": _to_float(fm.get("volume")),
        }

    def _fetch_spot(self) -> list:
        codes = ",".join(cfg["code"] for cfg in SPOT_MAP.values())
        data = self._get("/prices/latest", params={"by_code": codes})
        prices = (data.get("data") or {}).get("prices") or []
        by_code = {p["code"]: p for p in prices}

        rows = []
        for ticker, cfg in SPOT_MAP.items():
            p = by_code.get(cfg["code"])
            if p is None:
                continue
            ts = p.get("updated_at") or p.get("as_of") or p.get("created_at")
            if not ts:
                continue
            rows.append({
                "date": ts[:10],
                "ticker": ticker,
                "Open": None,
                "High": None,
                "Low": None,
                "Close": _to_float(p.get("price")),
                "Volume": None,
            })
        return rows

    def fetch_commodities(self):
        """Devuelve (df_futures, df_spot) en formato wide EOD.

        Index: DatetimeIndex. Columns: MultiIndex (field, ticker).
        """
        fut_rows = []
        for ticker in FUTURES_MAP:
            try:
                row = self._fetch_futures_front_month(ticker)
                if row is not None:
                    fut_rows.append(row)
            except Exception as e:
                print("  [WARN] futures " + ticker + ": " + str(e))

        try:
            spot_rows = self._fetch_spot()
        except Exception as e:
            print("  [WARN] spot: " + str(e))
            spot_rows = []

        return _rows_to_wide(fut_rows), _rows_to_wide(spot_rows)

    # --- Write ---------------------------------------------------

    def fetch_and_write(self, reference_date, run_id,
                        futures_path: str, spot_path: str) -> dict:
        """Fetch + append + write parquet + manifest.

        Devuelve dict con keys 'futures' y 'spot' (cada uno el manifest
        dict, o {} si fallo).
        """
        from src.utils import write_artifact_with_manifest

        df_fut, df_spot = self.fetch_commodities()
        result = {"futures": {}, "spot": {}}

        if not df_fut.empty:
            merged = _merge_with_existing(df_fut, futures_path)
            result["futures"] = write_artifact_with_manifest(
                merged, futures_path,
                source="oilpriceapi_futures",
                reference_date=reference_date,
                run_id=run_id,
                temporal_contract=None,
            )
        else:
            print("  [WARN] futures: sin datos nuevos")

        if not df_spot.empty:
            merged = _merge_with_existing(df_spot, spot_path)
            result["spot"] = write_artifact_with_manifest(
                merged, spot_path,
                source="oilpriceapi_spot",
                reference_date=reference_date,
                run_id=run_id,
                temporal_contract=None,
            )
        else:
            print("  [WARN] spot: sin datos nuevos")

        return result


# --- helpers a nivel de modulo -----------------------------------

def _to_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _rows_to_wide(rows: list) -> pd.DataFrame:
    """Convierte lista de dicts {date, ticker, Open, ...} a wide EOD.

    Index: DatetimeIndex. Columns: MultiIndex (field, ticker).
    """
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    wide = df.set_index(["date", "ticker"])[FIELDS].unstack("ticker")
    wide.columns = pd.MultiIndex.from_tuples(
        [(field, ticker) for field, ticker in wide.columns],
        names=["field", "ticker"],
    )
    wide.index.name = None
    wide = wide.sort_index()

    # K-FUTURES-DTYPE-01: normalizar FIELDS a numerico. El contrato FIELDS
    # declara que esas 5 magnitudes son numericas. Se recorre por columna
    # porque wide tiene MultiIndex (field, ticker) y wide[FIELDS] con lista
    # plana no aplica el contrato correctamente. Sin esto, columnas con
    # todos los valores None (Open/High/Low/Volume de spot) se infieren
    # como object y rompen el merge (FutureWarning pandas 2.x, fallo 3.x).
    # errors='coerce': valor no interpretable -> NaN (nunca imputar).
    for _col in wide.columns:
        if isinstance(_col, tuple) and _col[0] in FIELDS:
            wide[_col] = pd.to_numeric(wide[_col], errors='coerce')

    return wide


def _merge_with_existing(new_df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Merge new_df con el parquet existente, dedupe por fecha.

    Politica: la fila nueva gana sobre la previa (keep='last' tras
    concat en orden existing, new).
    """
    p = Path(path)
    if not p.exists():
        return new_df
    try:
        existing = pd.read_parquet(p)
    except Exception as e:
        print("  [WARN] no se pudo leer " + str(p) + ": " + str(e))
        return new_df

    # FU-009: evitar pd.concat con DataFrame vacio (FutureWarning pandas 2.x,
    # rotura en pandas 3.0).
    if existing is None or len(existing) == 0:
        return new_df

    combined = pd.concat([existing, new_df]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined
