import base64
import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from Crypto.Cipher import AES
from Crypto.Hash import MD5
from Crypto.Util.Padding import unpad


# =============================================================================
# Configuración
# =============================================================================

EURONEXT_ENDPOINT = "https://live.euronext.com/en/ajax/getHistoricalPricePopup/{instrument}"
EURONEXT_PASSWORD = "24ayqVo7yJma"
EURONEXT_MAP_PATH = Path("config/euronext_ticker_map.csv")
EURONEXT_CACHE_DIR = Path("data/cache/euronext")

EURONEXT_HEADERS = {
    "Accept": "*/*",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "User-Agent": "Mozilla/5.0",
}

# Columnas estándar que devuelve Euronext (en orden)
_CSV_COLUMNS = ["date", "open", "high", "low", "last", "close",
                "num_shares", "turnover", "vwap"]


# =============================================================================
# Utilidades criptográficas (CryptoJS AES reproducible)
# =============================================================================

def _evp_bytes_to_key(password: bytes, salt: bytes, key_len: int, iv_len: int):
    """EVP_BytesToKey con MD5, tal como lo usa CryptoJS."""
    derived = b""
    previous = b""
    while len(derived) < key_len + iv_len:
        md5 = MD5.new()
        md5.update(previous + password + salt)
        previous = md5.digest()
        derived += previous
    return derived[:key_len], derived[key_len:key_len + iv_len]


def _decrypt_payload(payload: dict) -> str:
    """Descifra el blob {ct, iv, s} devuelto por Euronext."""
    if not isinstance(payload, dict) or "ct" not in payload:
        raise ValueError(f"Payload inválido: {list(payload.keys()) if isinstance(payload, dict) else type(payload)}")

    ct = base64.b64decode(payload["ct"])
    salt = bytes.fromhex(payload["s"])
    key, derived_iv = _evp_bytes_to_key(EURONEXT_PASSWORD.encode(), salt, 32, 16)
    iv = bytes.fromhex(payload["iv"]) if payload.get("iv") else derived_iv

    cipher = AES.new(key, AES.MODE_CBC, iv)
    raw = unpad(cipher.decrypt(ct), AES.block_size).decode("utf-8")

    # El AES descifra a un string con escapes JSON (\n, \t, \", \/)
    try:
        return json.loads('"' + raw + '"')
    except json.JSONDecodeError:
        return (raw.replace(r"\/", "/")
                   .replace(r"\n", "\n")
                   .replace(r"\t", "\t")
                   .replace(r"\"", '"'))


# =============================================================================
# Parser HTML -> DataFrame OHLCV
# =============================================================================

def _parse_rows(html: str):
    """Extrae las filas <tr> del <tbody>. Devuelve lista de listas."""
    tbody_m = re.search(r"<tbody[^>]*>(.*?)</tbody>", html, re.IGNORECASE | re.DOTALL)
    if not tbody_m:
        return []

    rows = []
    for tr in re.finditer(r"<tr[^>]*>(.*?)</tr>", tbody_m.group(1), re.IGNORECASE | re.DOTALL):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr.group(1), re.IGNORECASE | re.DOTALL)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if cells and re.fullmatch(r"\d{2}/\d{2}/\d{4}", cells[0].strip()):
            rows.append(cells)
    return rows


def _rows_to_dataframe(rows) -> pd.DataFrame:
    """Convierte filas crudas a DataFrame OHLCV normalizado."""
    if not rows:
        return pd.DataFrame(columns=_CSV_COLUMNS)

    normalized = [r[:9] for r in rows if len(r) >= 9]
    if not normalized:
        return pd.DataFrame(columns=_CSV_COLUMNS)

    df = pd.DataFrame(normalized, columns=_CSV_COLUMNS)

    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")

    for col in _CSV_COLUMNS[1:]:
        df[col] = (df[col].astype(str)
                          .str.replace(",", "", regex=False)
                          .str.replace(" ", "", regex=False)
                          .str.strip())
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (df.dropna(subset=["date"])
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True))
    df.index.name = "Date"
    return df


# =============================================================================
# Provider
# =============================================================================

class EuronextProvider:
    """Proveedor Euronext para tickers europeos (PA / AS / MI).

    Uso:
        provider = EuronextProvider()
        df = provider.get_prices(["AIR.PA", "ASML.AS"], nb_session=500)

    Devuelve DataFrame MultiIndex compatible con el resto del sistema:
        ('Open', 'AIR.PA'), ('High', 'AIR.PA'), ...
    """

    name = "Euronext"

    def __init__(self):
        self._map = self._load_map()
        self._session = requests.Session()
        self._session.headers.update(EURONEXT_HEADERS)
        EURONEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------- Map --------------------

    def _load_map(self) -> dict:
        if not EURONEXT_MAP_PATH.exists():
            return {}
        df = pd.read_csv(EURONEXT_MAP_PATH)
        return {
            row["yahoo_ticker"]: {
                "euronext_id": row["euronext_id"],
                "isin": row["isin"],
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

    def _post(self, euronext_id: str, enddate: str, nb_session: int, retries: int = 3) -> str:
        """POST al endpoint de Euronext. Devuelve el HTML descifrado."""
        url = EURONEXT_ENDPOINT.format(instrument=euronext_id)
        data = [
            ("format", "csv"),
            ("decimal_separator", "."),
            ("date_form", "d/m/Y"),
            ("adjusted", "Y"),
            ("base100", ""),
            ("adjusted", "Y"),
            ("enddate", enddate),
            ("nbSession", str(nb_session)),
        ]

        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                r = self._session.post(url, data=data, timeout=30)
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")
                payload = r.json()
                return _decrypt_payload(payload)
            except Exception as e:
                last_exc = e
                if attempt < retries:
                    time.sleep(2 * attempt)
        raise RuntimeError(f"Euronext fallo tras {retries} intentos: {last_exc}")

    # -------------------- Cache --------------------

    def _cache_path(self, ticker: str) -> Path:
        return EURONEXT_CACHE_DIR / f"{ticker.replace('.', '_')}.csv"

    def _load_cache(self, ticker: str) -> pd.DataFrame:
        p = self._cache_path(ticker)
        if not p.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(p, parse_dates=["date"])
            return df
        except Exception:
            return pd.DataFrame()

    def _save_cache(self, ticker: str, df: pd.DataFrame):
        if df.empty:
            return
        p = self._cache_path(ticker)
        df.to_csv(p, index=False)

    def _cache_is_fresh(self, ticker: str) -> bool:
        """Considera fresco si el último dato es de hoy o del último día hábil."""
        df = self._load_cache(ticker)
        if df.empty:
            return False
        last = pd.to_datetime(df["date"]).max()
        return (pd.Timestamp.now().normalize() - last).days <= 1

    # -------------------- API pública --------------------

    def get_prices(self, tickers, nb_session: int = 500, use_cache: bool = True) -> pd.DataFrame:
        """Descarga OHLCV de los tickers indicados.

        Args:
            tickers: lista de tickers Yahoo (ej. ['AIR.PA', 'ASML.AS']).
            nb_session: número de sesiones a pedir (500 ~ 2 años).
            use_cache: si True, reutiliza cache fresca y evita peticiones.

        Returns:
            DataFrame MultiIndex ('Open', ticker), ('High', ticker), ...
        """
        frames = []
        enddate = datetime.now().strftime("%Y-%m-%d")

        for t in tickers:
            if not self.supports(t):
                print(f"  [EURONEXT] {t} no está en el mapa")
                continue

            # Cache fresca
            if use_cache and self._cache_is_fresh(t):
                df_cached = self._load_cache(t)
                if not df_cached.empty:
                    print(f"  [EURONEXT] {t} desde cache ({len(df_cached)} filas)")
                    frames.append(self._to_multiindex(df_cached, t))
                    continue

            info = self._map[t]
            euronext_id = info["euronext_id"]

            try:
                html = self._post(euronext_id, enddate, nb_session)
                rows = _parse_rows(html)
                df = _rows_to_dataframe(rows)

                if df.empty:
                    print(f"  [EURONEXT] {t} sin datos")
                    continue

                # Fusionar con cache previo
                df_prev = self._load_cache(t)
                if not df_prev.empty:
                    df = pd.concat([df_prev, df], ignore_index=True)
                    df = (df.drop_duplicates(subset=["date"])
                            .sort_values("date")
                            .reset_index(drop=True))

                self._save_cache(t, df)
                print(f"  [EURONEXT] {t} OK ({len(df)} filas, "
                      f"{df['date'].min().strftime('%Y-%m-%d')} -> "
                      f"{df['date'].max().strftime('%Y-%m-%d')})")
                frames.append(self._to_multiindex(df, t))

            except Exception as e:
                print(f"  [EURONEXT] {t} error: {e}")
                continue

        if not frames:
            return pd.DataFrame()

        data = pd.concat(frames, axis=1)
        if not isinstance(data.columns, pd.MultiIndex):
            data.columns = pd.MultiIndex.from_tuples(data.columns)
        return data

    def _to_multiindex(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Convierte DataFrame plano a MultiIndex compatible con Yahoo."""
        out = df[["date", "open", "high", "low", "close", "num_shares"]].copy()
        out = out.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "num_shares": "Volume",
        })
        out = out.set_index("date")
        out.index.name = "Date"  # <- mismo nombre que Yahoo para alinear
        out.columns = pd.MultiIndex.from_product([out.columns, [ticker]])
        return out
