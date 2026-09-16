"""Provider CBOE Index para indices de volatilidad (FU-021-3D).

Universo inicial:
  ^VIX3M (CBOE 3-Month Volatility Index)

Fuente: CBOE CDN publico, sin autenticacion.
  https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv

Prueba de aceptacion (2026-09-17, HEAD 18601ee):
  - HTTP 200, CSV con cabecera DATE,OPEN,HIGH,LOW,CLOSE
  - 4273 filas, rango 2009-09-18 -> 2026-09-15
  - 0 NaN en Close, 0 duplicados, ordenado
  - Reproducible byte-exacto entre fetches consecutivos
  - Inyeccion en df_market -> VOLATILITY_INDEX pasa de INSUFFICIENT
    a OK/STALE con coverage=1.0 (ver docs/auditoria/E5_INFORME_VIX3M.md)

Salida: data/cboe_vix3m.parquet + manifest FU-002.

Reglas:
  - Sin fallback a Yahoo (Yahoo dejo de servir historico de ^VIX3M).
  - Sin auth: no hay API key.
  - Reintentos: 1 en timeout. Sin reintento en 4xx.
  - Acumulacion historica: merge con parquet previo, dedupe por fecha.

Refs: FU-021-3D, R6, dictamen del auditor 2026-09-17.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from .base import MarketDataProvider


URL = ("https://cdn.cboe.com/api/global/us_indices/daily_prices/"
       "VIX3M_History.csv")
DEFAULT_TIMEOUT = 30
TICKER = "^VIX3M"
FIELDS = ["Open", "High", "Low", "Close", "Volume"]


class CboeIndexProvider(MarketDataProvider):
    """Descarga indices CBOE (CSV publico). Provider paralelo a Yahoo.

    No tiene relacion con data/providers/cboe.py::CboeProvider
    (scraper HTML de estadisticas de opciones). Dominios distintos,
    clases distintas, consumidores distintos.
    """

    name = "CBOE Index"

    def __init__(self):
        self.session = requests.Session()

    # --- HTTP con politica de reintentos -------------------------

    def _get_csv(self) -> str:
        attempts = 0
        while True:
            attempts += 1
            try:
                r = self.session.get(
                    URL,
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=DEFAULT_TIMEOUT,
                )
            except requests.exceptions.Timeout:
                if attempts >= 2:
                    raise
                continue

            if r.status_code >= 400:
                raise RuntimeError(
                    "CBOE " + str(r.status_code) + ": " + r.text[:200]
                )
            return r.text

    # --- Interfaz MarketDataProvider -----------------------------

    def get_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        try:
            r = self.session.head(URL, timeout=DEFAULT_TIMEOUT)
            return r.status_code == 200
        except Exception as e:
            print("CboeIndexProvider.is_available: " + str(e))
            return False

    def get_prices(self, tickers, start=None, end=None, period="10y"):
        raise NotImplementedError(
            "CboeIndexProvider no usa get_prices. Usar fetch_vix3m()."
        )

    def get_treasury_yields(self, maturities=None):
        raise NotImplementedError("CboeIndexProvider no provee yields.")

    def get_fed_data(self, series=None):
        raise NotImplementedError("CboeIndexProvider no provee datos Fed.")

    # --- Fetch ---------------------------------------------------

    def fetch_vix3m(self) -> pd.DataFrame:
        """Descarga y parsea el CSV de VIX3M. Devuelve wide EOD.

        Index: DatetimeIndex. Columns: MultiIndex (field, ticker).
        """
        try:
            text = self._get_csv()
        except Exception as e:
            print("  [WARN] CBOE fetch: " + str(e))
            return pd.DataFrame()
        return _parse_csv_to_wide(text, ticker=TICKER)

    # --- Write ---------------------------------------------------

    def fetch_and_write(self, reference_date, run_id,
                        parquet_path: str) -> dict:
        """Fetch + append + write parquet + manifest.

        Devuelve el manifest dict ({} si no hay datos nuevos).
        """
        from src.utils import write_artifact_with_manifest

        df = self.fetch_vix3m()
        if df.empty:
            print("  [WARN] cboe_vix3m: sin datos nuevos")
            return {}

        merged = _merge_with_existing(df, parquet_path)
        return write_artifact_with_manifest(
            merged, parquet_path,
            source="cboe_vix3m",
            reference_date=reference_date,
            run_id=run_id,
            temporal_contract=None,
        )


# --- helpers a nivel de modulo -----------------------------------

def _parse_csv_to_wide(csv_text: str, ticker: str) -> pd.DataFrame:
    """Parsea CSV CBOE (DATE,OPEN,HIGH,LOW,CLOSE) a wide MultiIndex.

    Volume no esta en el CSV: se agrega como columna NaN para
    mantener consistencia con el resto de df_market.
    """
    df = pd.read_csv(StringIO(csv_text))
    df.columns = [str(c).strip().upper() for c in df.columns]

    if "DATE" not in df.columns:
        raise ValueError("CSV CBOE sin columna DATE")

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])
    df = df.set_index("DATE")

    rename_map = {"OPEN": "Open", "HIGH": "High", "LOW": "Low", "CLOSE": "Close"}
    df = df.rename(columns=rename_map)

    for required in ("Open", "High", "Low", "Close"):
        if required not in df.columns:
            df[required] = float("nan")

    df["Volume"] = float("nan")
    df = df[FIELDS]

    df.columns = pd.MultiIndex.from_tuples(
        [(field, ticker) for field in df.columns],
        names=["field", "ticker"],
    )
    df.index.name = None
    df = df.sort_index()
    return df


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

    if existing is None or len(existing) == 0:
        return new_df

    combined = pd.concat([existing, new_df]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined