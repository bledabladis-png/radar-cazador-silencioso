"""Descarga la Official List of Section 13(f) de SEC.

URL canonica: https://www.sec.gov/files/investment/13flist{YYYY}q{Q}.txt
Excepcion historica: 2025Q3 -> 13flist2025q3-txt.txt

Destino: data/sec_13f/official_list_13f/13flist_{YYYYQn}.txt

Uso:
    py scripts/download_official_list_13f.py --quarter 2026Q1
    py scripts/download_official_list_13f.py --quarter 2026Q1 --force
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.institutional_accumulation.sec_13f.identity.sec13f_list import (
    parse_official_list_text,
)

SEC_BASE = "https://www.sec.gov/files/investment"
USER_AGENT = "Radar Sectorial Research <bledabladis@gmail.com>"
DEST_DIR = ROOT / "data" / "sec_13f" / "official_list_13f"

# Excepciones historicas conocidas
URL_EXCEPTIONS = {
    "2025Q3": "13flist2025q3-txt.txt",
}

# Sanity check: la lista oficial tiene >10.000 filas. Umbral bajo
# para tolerar variaciones, pero descartar ficheros vacios o HTML de error.
MIN_ROWS = 5000

def _parse_quarter(quarter):
    """2026Q1 -> ("2026", "1"). Raise ValueError si formato invalido."""
    if not isinstance(quarter, str) or len(quarter) != 6 or quarter[4] != "Q":
        raise ValueError(
            "quarter invalido: {!r} (esperado YYYYQn, p.ej. 2026Q1)".format(quarter))
    year, qn = quarter[:4], quarter[5:]
    if qn not in ("1", "2", "3", "4"):
        raise ValueError("trimestre invalido: {!r}".format(quarter))
    if not year.isdigit():
        raise ValueError("año invalido: {!r}".format(quarter))
    return year, qn


def _build_url(quarter):
    """Construye la URL canonica para un trimestre YYYYQn."""
    if quarter in URL_EXCEPTIONS:
        return SEC_BASE + "/" + URL_EXCEPTIONS[quarter]
    year, qn = _parse_quarter(quarter)
    return "{}/13flist{}q{}.txt".format(SEC_BASE, year, qn)


def _dest_path(quarter):
    return DEST_DIR / ("13flist_" + quarter + ".txt")


def _http_get(url):
    """GET con User-Agent identificable. Devuelve el cuerpo como bytes."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _validate_content(text):
    """Verifica que el fichero parsea. Devuelve num filas.

    Raise ValueError si el parsing falla o si el numero de filas es
    sospechosamente bajo (fichero corrupto o vacio).
    """
    df = parse_official_list_text(text)
    n = len(df)
    if n < MIN_ROWS:
        raise ValueError(
            "fichero sospechosamente corto: {} filas < {}".format(n, MIN_ROWS))
    return n


def download_official_list(quarter, *, force=False):
    """Descarga la Official List para un trimestre.

    Cache: si el destino existe y force=False, se reutiliza.
    Devuelve Path al fichero descargado.
    """
    dest = _dest_path(quarter)
    if dest.exists() and not force:
        print("[OK] cache hit: " + str(dest))
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = _build_url(quarter)
    print("[DOWNLOAD] " + url)

    body = _http_get(url)
    text = body.decode("utf-8")
    n = _validate_content(text)
    dest.write_text(text, encoding="utf-8", newline="\n")
    print("[OK] " + str(dest) + "  (" + str(n) + " filas)")
    return dest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarter", required=True,
                    help="Trimestre YYYYQn, p.ej. 2026Q1")
    ap.add_argument("--force", action="store_true",
                    help="Ignora cache local y redescarga")
    args = ap.parse_args()

    try:
        download_official_list(args.quarter, force=args.force)
    except urllib.error.HTTPError as e:
        print("[FAIL] HTTP {} en {}".format(e.code, e.url))
        return 1
    except urllib.error.URLError as e:
        print("[FAIL] URLError: {}".format(e))
        return 1
    except Exception as e:
        print("[ERROR] {}: {}".format(type(e).__name__, e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())