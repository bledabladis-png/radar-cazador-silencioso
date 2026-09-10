import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import websocket


# =============================================================================
# Configuración
# =============================================================================

XETRA_TOKEN_URL = "https://api.live.deutsche-boerse.com/v1/mdstokenservice/token"
XETRA_WS_URL = "wss://api.live.deutsche-boerse.com/v1/mds/ws"
XETRA_TRACING_SALT = "af5a8d16eb5dc49f8a72b26fd9185475c7a"
XETRA_MDS_TRACING_ID = "ea65e63f-b88f-414f-b1a9-e035263c8b0f"

XETRA_MAP_PATH = Path("config/xetra_ticker_map.csv")
XETRA_CACHE_DIR = Path("data/cache/xetra")

TOKEN_TTL_SAFETY = 120  # refresh token si quedan menos de 120s de vida
WS_RECV_TIMEOUT = 3.0
WS_QUERY_TIMEOUT = 25  # segundos máximos por consulta de un ticker
# Histórico inicial: ~300 sesiones (~430 días naturales).
# Suficiente para EMA200, RS126 y Wyckoff sin descargar años innecesarios.
from datetime import timedelta
WS_START = (datetime.now() - timedelta(days=430)).strftime("%Y-%m-%dT00:00:00.000Z")
WS_RESOLUTION = "1D"


# =============================================================================
# Utilidades criptográficas para el token
# =============================================================================

def _md5(v: str) -> str:
    return hashlib.md5(v.encode("utf-8")).hexdigest()


def _sha256(v: str) -> str:
    return hashlib.sha256(v.encode("utf-8")).hexdigest()


def _build_token_headers() -> dict:
    now_local = datetime.now().astimezone()
    now_utc = now_local.astimezone(timezone.utc)

    client_date = now_local.isoformat(timespec="seconds")
    request_datetime = now_utc.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    security_datetime = now_local.strftime("%Y%m%d%H%M")

    client_trace_id = _md5(client_date + XETRA_TOKEN_URL + XETRA_TRACING_SALT)
    pathname = "/mdstokenservice/token"
    request_trace_id = _sha256(pathname + "@" + request_datetime + "W" + XETRA_MDS_TRACING_ID)

    return {
        "Accept": "application/json, text/plain, */*",
        "Cache-Control": "no-cache",
        "Origin": "https://live.deutsche-boerse.com",
        "Pragma": "no-cache",
        "Referer": "https://live.deutsche-boerse.com/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/152.0.0.0 Safari/537.36"
        ),
        "Client-Date": client_date,
        "X-Client-TraceId": client_trace_id,
        "X-Security": _md5(security_datetime),
        "X-Request-Datetime": request_datetime,
        "X-Request-Trace-ID": request_trace_id,
    }


def _fetch_token(retries: int = 3) -> str:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(XETRA_TOKEN_URL, headers=_build_token_headers(), timeout=20)
            r.raise_for_status()
            data = r.json()
            token = data.get("token")
            if not token:
                raise RuntimeError(f"Sin token en respuesta: {data}")
            return token
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Fallo obteniendo token Xetra tras {retries} intentos: {last_exc}")


# =============================================================================
# XetraProvider
# =============================================================================

class XetraProvider:
    """Proveedor Deutsche Börse (Xetra) para tickers alemanes (.DE).

    Usa el WebSocket MDS público con autenticación JWT obtenida del endpoint
    /v1/mdstokenservice/token. Devuelve OHLCV histórico por ticker.

    Uso:
        provider = XetraProvider()
        df = provider.get_prices(["SAP.DE", "SIE.DE"])

    Salida: DataFrame MultiIndex ('Open', ticker), ('High', ticker), ...
    """

    name = "Xetra"

    def __init__(self):
        self._map = self._load_map()
        self._token = None
        self._token_expires_at = 0.0
        XETRA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------- Mapa --------------------

    def _load_map(self) -> dict:
        if not XETRA_MAP_PATH.exists():
            return {}
        df = pd.read_csv(XETRA_MAP_PATH)
        return {
            row["yahoo_ticker"]: {
                "isin": row["xetra_isin"],
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

    # -------------------- Token --------------------

    def _ensure_token(self) -> str:
        """Obtiene un token si no hay o está a punto de expirar."""
        now = time.time()
        if self._token and now < self._token_expires_at:
            return self._token
        self._token = _fetch_token()
        # Los tokens duran ~175s; refrescamos antes por seguridad
        self._token_expires_at = now + 120
        return self._token

    # -------------------- WebSocket --------------------

    def _open_ws(self):
        """Abre el WebSocket y autentica."""
        token = self._ensure_token()
        ws = websocket.create_connection(XETRA_WS_URL, timeout=15)
        ws.send(json.dumps({
            "subscribeAuthentication": {"token": token},
            "requestId": "auth"
        }))
        # Esperar confirmación de autenticación
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                ws.settimeout(3.0)
                raw = ws.recv()
            except Exception:
                continue
            try:
                d = json.loads(raw)
            except Exception:
                continue
            if d.get("requestId") == "auth":
                # Verificar que no hay error
                txt = json.dumps(d).lower()
                if "unauthorized" in txt or "fail" in txt:
                    ws.close()
                    raise RuntimeError(f"Autenticación Xetra fallida: {d}")
                return ws
        ws.close()
        raise RuntimeError("Xetra: no se recibió confirmación de autenticación")

    def _query_one(self, ws, req_id: str, isin: str, start: str, end: str) -> list:
        """Consulta un ticker y devuelve las filas."""
        fmt = f"DELAYED[{isin}@ETR>STX]"
        ws.send(json.dumps({
            "listTimeseries": {
                "resolution": WS_RESOLUTION,
                "marketstateId": fmt,
                "start": start,
                "end": end,
                "cleanSplits": False,
                "cleanDividends": False,
                "cleanDistributions": False,
                "cleanSubscriptions": False,
                "quality": "DELAYED"
            },
            "requestId": req_id
        }))

        rows = []
        deadline = time.time() + WS_QUERY_TIMEOUT
        while time.time() < deadline:
            try:
                ws.settimeout(WS_RECV_TIMEOUT)
                raw = ws.recv()
            except Exception:
                # No llega nada: seguir esperando hasta deadline
                continue
            if not raw:
                continue
            try:
                d = json.loads(raw)
            except Exception:
                continue
            if d.get("requestId") != req_id:
                continue
            ts = d.get("dataTimeseries")
            if ts is not None:
                if isinstance(ts, list):
                    rows.extend(ts)
                else:
                    rows.append(ts)
                continue
            if d.get("isComplete"):
                return rows
        return rows

    # -------------------- Cache --------------------

    def _cache_path(self, ticker: str) -> Path:
        return XETRA_CACHE_DIR / f"{ticker.replace('.', '_')}.csv"

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

    # -------------------- API pública --------------------

    def get_prices(self, tickers, use_cache: bool = True) -> pd.DataFrame:
        """Descarga OHLCV para los tickers indicados.

        Args:
            tickers: lista de tickers Yahoo (.DE).
            use_cache: si True, reutiliza cache fresca.

        Returns:
            DataFrame MultiIndex compatible con el resto del sistema.
        """
        if not tickers:
            return pd.DataFrame()

        frames = []
        pending = []

        # Filtrar por cache primero
        for t in tickers:
            if not self.supports(t):
                print(f"  [XETRA] {t} no está en el mapa")
                continue
            if use_cache and self._cache_is_fresh(t):
                df_cached = self._load_cache(t)
                if not df_cached.empty:
                    print(f"  [XETRA] {t} desde cache ({len(df_cached)} filas)")
                    frames.append(self._to_multiindex(df_cached, t))
                    continue
            pending.append(t)

        if not pending:
            return self._concat_frames(frames)

        # Abrir WebSocket para los pendientes
        end_iso = datetime.now().strftime("%Y-%m-%dT23:59:00.000Z")
        try:
            ws = self._open_ws()
        except Exception as e:
            print(f"  [XETRA] No se pudo abrir WebSocket: {e}")
            return self._concat_frames(frames)

        try:
            for i, t in enumerate(pending, 1):
                info = self._map[t]
                isin = info["isin"]
                try:
                    rows = self._query_one(ws, f"q{i}", isin, WS_START, end_iso)
                    if not rows:
                        print(f"  [XETRA] {t} sin datos")
                        continue
                    df_new = self._rows_to_df(rows)

                    # Fusionar con cache previo
                    df_prev = self._load_cache(t)
                    if not df_prev.empty:
                        df_new = pd.concat([df_prev, df_new], ignore_index=True)
                        df_new = (df_new.drop_duplicates(subset=["date"])
                                        .sort_values("date")
                                        .reset_index(drop=True))

                    self._save_cache(t, df_new)
                    print(f"  [XETRA] {t} OK ({len(df_new)} filas, "
                          f"{df_new['date'].min().strftime('%Y-%m-%d')} -> "
                          f"{df_new['date'].max().strftime('%Y-%m-%d')})")
                    frames.append(self._to_multiindex(df_new, t))
                except Exception as e:
                    print(f"  [XETRA] {t} error: {e}")
                    continue
        finally:
            try:
                ws.close()
            except Exception:
                pass

        return self._concat_frames(frames)

    # -------------------- Helpers --------------------

    def _rows_to_df(self, rows) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if "date" not in df.columns:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in ["open", "high", "low", "close", "quantity", "turnover"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = (df.dropna(subset=["date"])
                .drop_duplicates(subset=["date"])
                .sort_values("date")
                .reset_index(drop=True))
        return df

    def _to_multiindex(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        out = df[["date", "open", "high", "low", "close", "quantity"]].copy()
        out = out.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "quantity": "Volume",
        })
        out = out.set_index("date")
        out.index.name = "Date"
        out.columns = pd.MultiIndex.from_product([out.columns, [ticker]])
        return out

    def _concat_frames(self, frames) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()
        data = pd.concat(frames, axis=1)
        if not isinstance(data.columns, pd.MultiIndex):
            data.columns = pd.MultiIndex.from_tuples(data.columns)
        if data.columns.duplicated().any():
            data = data.loc[:, ~data.columns.duplicated(keep="last")]
        return data
