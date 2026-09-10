"""
Verificacion del endpoint BME (HistoricalSharesPrices) para acciones.
Se ejecuta localmente y desde GitHub Actions. NO modifica el sistema.
"""
import json
import sys
import time
from datetime import datetime

import requests


BASE_URL = "https://apiweb.bolsasymercados.es/Market/v1/EQ/HistoricalSharesPrices"

HEADERS_FULL = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/152.0.0.0 Safari/537.36"
    ),
    "Origin": "https://www.bolsamadrid.es",
    "Referer": "https://www.bolsamadrid.es/",
}

HEADERS_MIN = {
    "User-Agent": "Mozilla/5.0",
}


def fetch(isin, date_from, date_to, page=0, page_size=0, headers=None, label=""):
    if headers is None:
        headers = HEADERS_FULL
    params = {
        "type": "V",
        "isin": isin,
        "from": date_from,
        "to": date_to,
        "page": page,
        "pageSize": page_size,
    }
    print("")
    print("=" * 60)
    print("TEST: " + label)
    print("=" * 60)
    print("URL:   " + BASE_URL)
    print("Params: " + str(params))
    print("Headers: " + str(list(headers.keys())))
    t0 = time.time()
    try:
        r = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
        elapsed = time.time() - t0
        print("HTTP " + str(r.status_code) + " en " + str(round(elapsed, 2)) + "s")
        print("Content-Type: " + r.headers.get("Content-Type", "N/A"))
        print("Content-Length: " + r.headers.get("Content-Length", "N/A"))
        if r.status_code != 200:
            print("Body (first 500):")
            print(r.text[:500])
            return None
        try:
            data = r.json()
        except Exception as e:
            print("JSON parse error: " + str(e))
            print("Body (first 500): " + r.text[:500])
            return None
        # Extraer campos utiles
        total = data.get("totalResults", "N/A")
        has_more = data.get("hasMoreResults", "N/A")
        print("totalResults: " + str(total))
        print("hasMoreResults: " + str(has_more))
        # Buscar el array de registros
        records = None
        for key in ("details", "data", "sharesPrices", "items", "results"):
            if isinstance(data.get(key), list):
                records = data[key]
                print("records key: '" + key + "' (count=" + str(len(records)) + ")")
                break
        if records is None:
            # Buscar cualquier lista no vacia
            for k, v in data.items():
                if isinstance(v, list) and v:
                    records = v
                    print("records key (auto): '" + k + "' (count=" + str(len(v)) + ")")
                    break
        if records is None:
            print("No se encontro lista de registros. Keys: " + str(list(data.keys())))
            return data
        if records:
            print("Primer registro:")
            print(json.dumps(records[0], indent=2, ensure_ascii=False)[:600])
            print("Ultimo registro (resumen):")
            last = records[-1]
            print("  date=" + str(last.get("date")) + "  close=" + str(last.get("close")))
        else:
            print("Lista vacia.")
        return data
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("[FAIL] Excepcion: " + str(e))
        return None


def main():
    print("Entorno: GitHub Actions = " + str(__import__("os").environ.get("GITHUB_ACTIONS", "no")))
    print("Python: " + sys.version)
    print("Timestamp: " + datetime.now().isoformat())

    results = {}

    # T1 - Repsol rango largo
    results["T1_REP_2y"] = fetch(
        "ES0173516115", "20240101", "20260901",
        label="T1: Repsol (REP) 2024-01-01 a 2026-09-01 (pageSize=0)"
    )

    # T2 - Inditex rango largo
    results["T2_ITX_2y"] = fetch(
        "ES0148396007", "20240101", "20260901",
        label="T2: Inditex (ITX) 2024-01-01 a 2026-09-01 (pageSize=0)"
    )

    # T3 - Repsol ventana corta (10 dias)
    results["T3_REP_short"] = fetch(
        "ES0173516115", "20260825", "20260905",
        label="T3: Repsol ventana corta (10 dias)"
    )

    # T4 - Repsol con pageSize=100
    results["T4_REP_ps100"] = fetch(
        "ES0173516115", "20240101", "20260901", page_size=100,
        label="T4: Repsol pageSize=100"
    )

    # T5 - ISIN invalido
    results["T5_INVALID"] = fetch(
        "XX0000000000", "20240101", "20260901",
        label="T5: ISIN invalido XX0000000000"
    )

    # T6 - Sin Origin/Referer (solo User-Agent)
    results["T6_MIN_HEADERS"] = fetch(
        "ES0173516115", "20240101", "20260901", headers=HEADERS_MIN,
        label="T6: Repsol con headers minimos (sin Origin/Referer)"
    )

    # Resumen final
    print("")
    print("=" * 60)
    print("RESUMEN")
    print("=" * 60)
    for name, data in results.items():
        if data is None:
            print("  " + name + ": FAIL (sin respuesta)")
        else:
            total = data.get("totalResults", "?")
            print("  " + name + ": OK (totalResults=" + str(total) + ")")

    any_ok = any(d is not None for d in results.values())
    sys.exit(0 if any_ok else 1)


if __name__ == "__main__":
    main()
