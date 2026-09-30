import hashlib
import json
import time
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from .base import MarketDataProvider

BASE_URL = "https://api.finra.org/data/group/otcMarket/name"

# Cache local de respuestas paginadas. Los datos FINRA son semanales
# e inmutables una vez publicados; se cachean 30 dias. Los resultados
# vacios (semana aun no publicada) se cachean 24h para no machacar
# la API con reintentos inutiles.
_CACHE_DIR = Path("data/cache/finra")
_CACHE_TTL_HIT = 30 * 86400
_CACHE_TTL_EMPTY = 24 * 3600
_LATEST_WEEK_TTL = 300  # F5.7-05: TTL de memoize de get_latest_week (s).

class FinraProvider(MarketDataProvider):
    def __init__(self):
        self.name = "FINRA ATS"
        self._session = requests.Session()
        # F5.7-05: memoize de get_latest_week().
        # get_latest_week() itera hasta 6 semanas hacia atras, cada una
        # con su propia peticion HTTP. Dentro de un mismo run se llama
        # desde darkpool.py y darkpool_history.py con la misma instancia
        # (darkpool pasa la instancia a _backfill_history); sin memoize
        # el resultado se recalcula aunque sea identico.
        self._latest_week_cache = None
        self._latest_week_ts = 0.0
        self._session.headers.update({
            "Accept": "text/plain",
            "Content-Type": "application/json",
            "Origin": "https://otctransparency.finra.org",
            "Referer": "https://otctransparency.finra.org/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
        })

    def get_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        try:
            week = self.get_latest_week()
            return week is not None
        except (requests.RequestException, ValueError, KeyError, TypeError):
            return False

    # ------------------------------------------------------------
    # CACHE LOCAL
    # ------------------------------------------------------------
    def _cache_key(self, endpoint, payload):
        """Hash estable del payload (sin offset/limit que cambian por pagina)."""
        clean = {k: v for k, v in payload.items() if k not in ("offset", "limit")}
        blob = json.dumps({"endpoint": endpoint, "payload": clean},
                          sort_keys=True, default=str)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
        return f"{endpoint}_{digest}"

    def _load_cache(self, key):
        """Devuelve DataFrame si hay cache valida, None si no.

        - Hit real: parquet con TTL 30d.
        - Miss marcado: fichero .empty con TTL 24h (resultado vacio reciente).
        """
        cache_dir = _CACHE_DIR
        parquet = cache_dir / f"{key}.parquet"
        empty = cache_dir / f"{key}.empty"
        now = time.time()
        if parquet.exists():
            age = now - parquet.stat().st_mtime
            if age < _CACHE_TTL_HIT:
                try:
                    return pd.read_parquet(parquet)
                except (OSError, ValueError, TypeError, pd.errors.ParserError):
                    parquet.unlink(missing_ok=True)
        if empty.exists():
            age = now - empty.stat().st_mtime
            if age < _CACHE_TTL_EMPTY:
                return pd.DataFrame()
        return None

    def _save_cache(self, key, df):
        cache_dir = _CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"{key}.parquet"
        tmp = target.with_suffix(".parquet.tmp")
        try:
            df.to_parquet(tmp, index=False)
            tmp.replace(target)
        except (OSError, ValueError, TypeError, KeyError) as e:
            print(f"  [finra] no se pudo escribir cache {target.name}: {e}")
            tmp.unlink(missing_ok=True)

    def _save_empty_marker(self, key):
        cache_dir = _CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        marker = cache_dir / f"{key}.empty"
        try:
            marker.write_text("", encoding="utf-8")
        except OSError as e:
            print(f"  [finra] no se pudo escribir marker {marker.name}: {e}")

    # ------------------------------------------------------------
    # PETICIONES
    # ------------------------------------------------------------
    def _post(self, endpoint, payload):
        try:
            resp = self._session.post(f"{BASE_URL}/{endpoint}", json=payload, timeout=60)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            print(f"ERROR en {endpoint}: {e}")
            if 'resp' in locals():
                print(f"Status: {resp.status_code}")
                print(resp.text[:500])
            return None

    def _paginated_request(self, endpoint, payload):
        key = self._cache_key(endpoint, payload)
        cached = self._load_cache(key)
        if cached is not None:
            return cached
        offset = 0
        frames = []
        while True:
            payload["offset"] = offset
            payload["limit"] = 5000
            resp = self._post(endpoint, payload)
            if resp is None:
                break
            # Leer CSV protegiendonos contra respuesta vacia
            try:
                df = pd.read_csv(StringIO(resp.text), sep="|", on_bad_lines="skip")
            except pd.errors.EmptyDataError:
                break
            if df.empty:
                break
            # Quedarnos solo con las columnas que nos interesan
            if "issueSymbolIdentifier" in df.columns and "totalWeeklyShareQuantity" in df.columns:
                df = df[["issueSymbolIdentifier", "totalWeeklyShareQuantity"]]
            frames.append(df)
            total = int(resp.headers.get("record-total", 0))
            offset += 5000
            if offset >= total:
                break
            time.sleep(2)
        if frames:
            result = pd.concat(frames, ignore_index=True)
            self._save_cache(key, result)
            return result
        self._save_empty_marker(key)
        return pd.DataFrame()

    # ------------------------------------------------------------
    # MÉTODOS PÚBLICOS
    # ------------------------------------------------------------
    def get_latest_week(self):
        # F5.7-05: memoize por instancia con TTL 300s.
        # Evita repetir la busqueda (hasta 6 semanas x 1 request) cuando
        # la misma instancia se reutiliza dentro del mismo run.
        now = time.time()
        if now - self._latest_week_ts < _LATEST_WEEK_TTL and self._latest_week_ts > 0:
            return self._latest_week_cache
        for i in range(6):
            test_date = (datetime.now() - timedelta(weeks=i))
            monday = test_date - timedelta(days=test_date.weekday())
            monday_str = monday.strftime('%Y-%m-%d')
            data = self.get_week_summary(monday_str)
            if not data.empty:
                self._latest_week_cache = monday_str
                self._latest_week_ts = now
                return monday_str
        self._latest_week_cache = None
        self._latest_week_ts = now
        return None

    def get_week_summary(self, week, tier="T1"):
        payload = {
            "quoteValues": False,
            "delimiter": "|",
            "limit": 5000,
            "fields": [
                "tierDescription", "issueSymbolIdentifier", "issueName",
                "marketParticipantName", "MPID", "totalWeeklyShareQuantity",
                "totalWeeklyTradeCount", "lastUpdateDate"
            ],
            "sortFields": ["issueSymbolIdentifier", "MPID"],
            "compareFilters": [
                {"fieldName": "summaryTypeCode", "fieldValue": "ATS_W_SMBL_FIRM", "compareType": "EQUAL"},
                {"fieldName": "weekStartDate", "fieldValue": week, "compareType": "EQUAL"},
                {"fieldName": "tierIdentifier", "fieldValue": tier, "compareType": "EQUAL"}
            ]
        }
        return self._paginated_request("weeklySummary", payload)

    def get_symbol(self, symbol, week, tier="T1"):
        payload = {
            "quoteValues": False,
            "delimiter": "|",
            "limit": 5000,
            "sortFields": ["-totalWeeklyShareQuantity"],
            "compareFilters": [
                {"fieldName": "summaryTypeCode", "fieldValue": "ATS_W_SMBL_FIRM", "compareType": "EQUAL"},
                {"fieldName": "issueSymbolIdentifier", "fieldValue": symbol, "compareType": "EQUAL"},
                {"fieldName": "weekStartDate", "fieldValue": week, "compareType": "EQUAL"},
                {"fieldName": "tierIdentifier", "fieldValue": tier, "compareType": "EQUAL"}
            ]
        }
        return self._paginated_request("weeklySummary", payload)

    def get_all_tiers(self, week):
        frames = []
        for tier in ["T1", "T2", "OTCE"]:
            df = self.get_week_summary(week, tier)
            if not df.empty:
                frames.append(df)
            time.sleep(2)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ------------------------------------------------------------
    # MÉTODOS NO IMPLEMENTADOS (interfaz)
    # ------------------------------------------------------------
    def get_prices(self, tickers, start=None, end=None, period=None):
        raise NotImplementedError("FINRA no proporciona precios")
    def get_treasury_yields(self, maturities=None, index=None):
        raise NotImplementedError("FINRA no proporciona yields")
    def get_fed_data(self, index=None):
        raise NotImplementedError("FINRA no proporciona datos de la Fed")
    def get_options_data(self, index=None):
        raise NotImplementedError("FINRA no proporciona datos de opciones")
