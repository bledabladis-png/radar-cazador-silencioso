"""
bme_provider.py -- Proveedor Bolsa de Madrid (BME).

Endpoint: https://apiweb.bolsasymercados.es/Market/v1/EQ/HistoricalSharesPrices
Sin autenticacion. JSON limpio. Verificado 2026-09-11 en GitHub Actions.

Mapeo BME -> OHLCV interno:
    Date   <- date       (YYYYMMDD)
    Open   <- reference  (precio referencia del dia; aproximacion de apertura)
    High   <- high
    Low    <- low
    Close  <- close
    Volume <- volume

Nota: BME no da Open real de subasta. Se usa "reference", que suele coincidir
con el cierre previo y actua como precio de referencia de la sesion.
"""
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

BME_ENDPOINT = "https://apiweb.bolsasymercados.es/Market/v1/EQ/HistoricalSharesPrices"
BME_MAP_PATH = Path("config/bme_ticker_map.csv")
BME_CACHE_DIR = Path("data/cache/bme")
BME_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/152.0.0.0 Safari/537.36"
    ),
}

MAX_SESSIONS = 1250
INTER_REQUEST_DELAY = 0.3

_CSV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


class BMEProvider:
    """Proveedor BME para tickers .MC (Bolsa de Madrid)."""

    name = "BME"

    def __init__(self):
        self._map = self._load_map()
        self._session = requests.Session()
        self._session.headers.update(BME_HEADERS)
        BME_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------- Mapa --------------------

    def _load_map(self) -> dict:
        if not BME_MAP_PATH.exists():
            return {}
        df = pd.read_csv(BME_MAP_PATH)
        return {
            row["yahoo_ticker"]: {
                "isin": row["bme_isin"],
                "mic": row["mic"],
                "exchange": row["exchange"],
                "country": row["country"],
            }
            for _, row in df.iterrows()
        }

    def supports(self, ticker: str) -> bool:
        return ticker in self._map

    def supported_tickers(self):
        return list(self._map.keys())

    # -------------------- HTTP --------------------

    def _fetch(self, isin: str, date_from: str, date_to: str, retries: int = 3) -> list:
        params = {
            "type": "V",
            "isin": isin,
            "from": date_from,
            "to": date_to,
            "page": 0,
            "pageSize": 0,
        }
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                r = self._session.get(BME_ENDPOINT, params=params, timeout=30)
                if r.status_code != 200:
                    raise RuntimeError("HTTP " + str(r.status_code))
                data = r.json()
                return data.get("data", [])
            except Exception as e:
                last_exc = e
                if attempt < retries:
                    time.sleep(2 * attempt)
        raise RuntimeError("BME fallo tras " + str(retries) + " intentos: " + str(last_exc))

    # -------------------- Cache --------------------

    def _cache_path(self, ticker: str) -> Path:
        return BME_CACHE_DIR / (ticker.replace(".", "_") + ".csv")

    def _load_cache(self, ticker: str) -> pd.DataFrame:
        p = self._cache_path(ticker)
        if not p.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(p, parse_dates=["date"])
        except Exception:
            return pd.DataFrame()

    def _save_cache(self, ticker: str, df: pd.DataFrame):
        if df.empty:
            return
        self._cache_path(ticker).write_bytes(df.to_csv(index=False).encode("utf-8"))

    def _cache_is_fresh(self, ticker: str) -> bool:
        df = self._load_cache(ticker)
        if df.empty:
            return False
        last = pd.to_datetime(df["date"]).max()
        return (pd.Timestamp.now().normalize() - last).days <= 1

    # -------------------- Conversion --------------------

    def _rows_to_df(self, rows) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=_CSV_COLUMNS)
        records = []
        for r in rows:
            try:
                date_str = str(r.get("date", ""))
                if len(date_str) != 8:
                    continue
                dt = datetime.strptime(date_str, "%Y%m%d")
                records.append({
                    "date": dt,
                    "open": r.get("reference"),
                    "high": r.get("high"),
                    "low": r.get("low"),
                    "close": r.get("close"),
                    "volume": r.get("volume"),
                })
            except Exception:
                continue
        if not records:
            return pd.DataFrame(columns=_CSV_COLUMNS)
        df = pd.DataFrame(records)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = (df.dropna(subset=["date"])
                .drop_duplicates(subset=["date"])
                .sort_values("date")
                .reset_index(drop=True))
        df.index.name = "Date"
        return df

    def _to_multiindex(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        out = df[["date", "open", "high", "low", "close", "volume"]].copy()
        out = out.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })
        out = out.set_index("date")
        out.index.name = "Date"
        out.columns = pd.MultiIndex.from_product([out.columns, [ticker]])
        return out

    # -------------------- API publica --------------------

    def get_prices(self, tickers, since_date=None, use_cache: bool = True) -> pd.DataFrame:
        """Descarga OHLCV de los tickers .MC via BME.

        Args:
            tickers: lista de tickers Yahoo (.MC).
            since_date: opcional, fecha ISO 'YYYY-MM-DD' minima a descargar.
            use_cache: si True, reutiliza cache fresca.
        """
        frames = []
        today = datetime.now()
        default_from = (today - timedelta(days=430)).strftime("%Y%m%d")

        for t in tickers:
            if not self.supports(t):
                print("  [BME] " + t + " no esta en el mapa")
                continue

            # Cache fresca -> skip
            if use_cache and self._cache_is_fresh(t):
                df_cached = self._load_cache(t)
                if not df_cached.empty:
                    print("  [BME] " + t + " desde cache (" + str(len(df_cached)) + " filas)")
                    frames.append(self._to_multiindex(df_cached, t))
                    continue

            # === Auto-recuperacion ===
            date_from = None
            if since_date is not None:
                date_from = since_date.replace("-", "")
            elif use_cache:
                df_cached = self._load_cache(t)
                if not df_cached.empty:
                    last_date = pd.to_datetime(df_cached["date"]).max()
                    days_gap = (pd.Timestamp.now().normalize() - last_date).days
                    if days_gap > 7:
                        print("  [BME] CACHE VIEJA: " + t + " sin datos desde " + str(last_date.date()) + " (" + str(days_gap) + " dias)")
                    date_from = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")
                    print("  [BME] " + t + " gap=" + str(days_gap) + "d, recuperando desde " + date_from)

            if date_from is None:
                date_from = default_from

            date_to = today.strftime("%Y%m%d")
            isin = self._map[t]["isin"]

            try:
                rows = self._fetch(isin, date_from, date_to)
                df_new = self._rows_to_df(rows)

                if df_new.empty:
                    print("  [BME] " + t + " sin datos")
                    continue

                df_prev = self._load_cache(t)
                if not df_prev.empty:
                    df_new = pd.concat([df_prev, df_new], ignore_index=True)
                    df_new = (df_new.drop_duplicates(subset=["date"])
                                    .sort_values("date")
                                    .reset_index(drop=True))

                self._save_cache(t, df_new)
                print("  [BME] " + t + " OK (" + str(len(df_new)) + " filas, "
                      + df_new["date"].min().strftime("%Y-%m-%d") + " -> "
                      + df_new["date"].max().strftime("%Y-%m-%d") + ")")
                frames.append(self._to_multiindex(df_new, t))
                time.sleep(INTER_REQUEST_DELAY)

            except Exception as e:
                print("  [BME] " + t + " error: " + str(e))
                continue

        if not frames:
            return pd.DataFrame()

        data = pd.concat(frames, axis=1)
        if not isinstance(data.columns, pd.MultiIndex):
            data.columns = pd.MultiIndex.from_tuples(data.columns)
        return data
