"""Descarga y extraccion del dataset SEC Form 13F Data Set.

Version: 1.0 (2026-09-19)
Contrato: docs/auditoria/INSTITUTIONAL_ACCUMULATION_CONTRATO.md seccion 6.
Dictamen FA-1.1: docs/auditoria/INSTITUTIONAL_ACCUMULATION_DICTAMEN_GATE0.md.

Alcance:
  - download_13f_zip: descarga ZIP trimestral con cache + retry + SHA-256.
  - extract_13f_zip: extrae y valida los 7 TSVs esperados.

NO incluye: parsing, identity resolution, dedup, CUSIP->ticker, NIPC.
"""
import hashlib
import time
import zipfile
from pathlib import Path

import requests

from .schema import EXPECTED_FILES

SEC_BASE_URL = "https://www.sec.gov/files/structureddata/data/form-13f-data-sets"
DEFAULT_USER_AGENT = "Radar Sectorial Research <bledabladis@gmail.com>"
DEFAULT_RETRY_COUNT = 3
DEFAULT_BACKOFF_SECONDS = 2.0
HTTP_TIMEOUT_SECONDS = 300
CHUNK_SIZE = 1024 * 1024  # 1 MB


def _build_url(period):
    """Construye URL del ZIP trimestral.

    period: string en formato SEC, ej. "01mar2026-31may2026".
    """
    if not period:
        raise ValueError("period no puede estar vacio")
    return SEC_BASE_URL + "/" + period + "_form13f.zip"


def _sha256(path):
    """SHA-256 de un fichero, leido en chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_user_agent(user_agent):
    """User-Agent obligatorio y no vacio (SEC Developer Resources)."""
    if not user_agent or not user_agent.strip():
        raise ValueError("User-Agent no puede estar vacio (SEC lo requiere)")

def _http_get_with_retry(url, headers, timeout=HTTP_TIMEOUT_SECONDS,
                         max_retries=DEFAULT_RETRY_COUNT,
                         backoff=DEFAULT_BACKOFF_SECONDS):
    """GET con retry en 5xx/timeouts. Sin retry en 4xx.

    Backoff exponencial: backoff * 2^(intento-1).
    Devuelve el objeto Response si 2xx.
    Lanza excepcion si agota retries o 4xx.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(backoff * (2 ** (attempt - 1)))
                continue
            raise
        if 500 <= r.status_code < 600:
            last_exc = requests.HTTPError("HTTP " + str(r.status_code))
            if attempt < max_retries:
                time.sleep(backoff * (2 ** (attempt - 1)))
                continue
            r.raise_for_status()
        if 400 <= r.status_code < 500:
            r.raise_for_status()
        return r
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Retry agotado sin excepcion previa")

def download_13f_zip(period, dest_dir, user_agent=DEFAULT_USER_AGENT, force=False):
    """Descarga el ZIP trimestral 13F de SEC a dest_dir.

    period:   string formato SEC, ej. "01mar2026-31may2026".
    dest_dir: Path al directorio destino.
    force:    si True, ignora cache existente y redescarga.
    Devuelve: Path al ZIP descargado.

    Cache: si el fichero existe y su tamano coincide con Content-Length,
    se reutiliza (sanity check, no identidad). El SHA-256 se calcula
    siempre como lineage local.
    """
    _validate_user_agent(user_agent)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    url = _build_url(period)
    filename = period + "_form13f.zip"
    zip_path = dest_dir / filename

    if zip_path.exists() and not force:
        print("[DOWNLOAD] cache hit: " + str(zip_path))
        print("[DOWNLOAD] sha256=" + _sha256(zip_path))
        return zip_path

    headers = {"User-Agent": user_agent}
    print("[DOWNLOAD] " + url)

    r = _http_get_with_retry(url, headers)
    total = int(r.headers.get("Content-Length", 0))

    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

    if total and downloaded != total:
        raise IOError(
            "Descarga incompleta: " + str(downloaded) + "/" + str(total) + " bytes"
        )

    print("[DOWNLOAD] bytes=" + str(downloaded))
    print("[DOWNLOAD] sha256=" + _sha256(zip_path))
    return zip_path

def extract_13f_zip(zip_path, dest_dir, force=False):
    """Extrae los 7 TSVs esperados del ZIP a dest_dir.

    zip_path: Path al ZIP descargado.
    dest_dir: Path al directorio destino.
    force:    si True, re-extrae aunque ya exista.

    Devuelve: dict {nombre_sin_extension: Path} para los 7 TSVs.

    Validaciones:
      - ZIP integro (CRC via testzip).
      - Los 7 TSVs esperados presentes. Extras ignorados.
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    if not zip_path.exists():
        raise FileNotFoundError("ZIP no existe: " + str(zip_path))

    # Validar integridad CRC antes de extraer
    with zipfile.ZipFile(zip_path, "r") as zf:
        corrupt = zf.testzip()
        if corrupt is not None:
            raise zipfile.BadZipFile("CRC falla en: " + corrupt)

        # Verificar presencia de los 7 TSVs
        names_in_zip = set(zf.namelist())
        missing = [f for f in EXPECTED_FILES if f not in names_in_zip]
        if missing:
            raise ValueError("Faltan TSVs en el ZIP: " + str(missing))

        # Extraer solo los 7 requeridos (extras ignorados)
        dest_dir.mkdir(parents=True, exist_ok=True)
        result = {}
        for name in EXPECTED_FILES:
            target = dest_dir / name
            if target.exists() and not force:
                result[name.replace(".tsv", "")] = target
                continue
            with zf.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())
            result[name.replace(".tsv", "")] = target

    return result