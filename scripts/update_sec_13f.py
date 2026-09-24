"""Orquestador de ingesta SEC 13F: descarga dataset + official list.

Uso:
    py scripts/update_sec_13f.py --quarter 2026Q2
    py scripts/update_sec_13f.py --latest
    py scripts/update_sec_13f.py --quarter 2026Q2 --force

Idempotente: si el parquet del trimestre ya existe y force=False, se
salta la ingesta. La official list siempre se descarga si falta.

No modifica el radar. Solo prepara los datos que `compute_iae_section`
consumira en el proximo `run.py`.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.institutional_accumulation.sec_13f.ingest import ingest_13f
from src.institutional_accumulation.sec_13f.downloader import _pick_base_url
from scripts.download_official_list_13f import download_official_list

# Fichero de 2 bytes que identifica el ultimo trimestre ingestado.
# Versionado en git. Usado por daily_run.yml para invalidar la cache
# de parquets 13F cuando aparece un trimestre nuevo.
LATEST_FILE = ROOT / "data" / "sec_13f" / "latest_quarter.txt"

# Meses del formato SEC (ingles, 3 letras):
_MONTHS = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
    7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec",
}

# Trimestre -> mes de inicio del period SEC (1-based):
# Q1 cierra 31-mar, dataset cubre 01-mar a 31-may
# Q2 cierra 30-jun, dataset cubre 01-jun a 31-aug
# Q3 cierra 30-sep, dataset cubre 01-sep a 30-nov
# Q4 cierra 31-dec, dataset cubre 01-dec a 28/29-feb (año+1)
_QUARTER_START_MONTH = {"Q1": 3, "Q2": 6, "Q3": 9, "Q4": 12}

_DAYS_IN_MONTH = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                  7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

def _is_leap(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def _last_day_of_month(year, month):
    if month == 2:
        return 29 if _is_leap(year) else 28
    return _DAYS_IN_MONTH[month]


def _parse_quarter(quarter):
    """YYYYQn -> (year:int, qn:str 'Q1'..'Q4'). Raise ValueError."""
    if not isinstance(quarter, str) or len(quarter) != 6 or quarter[4] != "Q":
        raise ValueError("quarter invalido: {!r} (esperado YYYYQn)".format(quarter))
    year_s, qn = quarter[:4], "Q" + quarter[5]
    if not year_s.isdigit():
        raise ValueError("año invalido: {!r}".format(quarter))
    if qn not in ("Q1", "Q2", "Q3", "Q4"):
        raise ValueError("trimestre invalido: {!r}".format(quarter))
    return int(year_s), qn


def quarter_to_iso_end(quarter):
    """YYYYQn -> fecha ISO del cierre del trimestre (YYYY-MM-DD)."""
    year, qn = _parse_quarter(quarter)
    end_month = {"Q1": 3, "Q2": 6, "Q3": 9, "Q4": 12}[qn]
    end_day = _last_day_of_month(year, end_month)
    return "{:04d}-{:02d}-{:02d}".format(year, end_month, end_day)


def quarter_to_source_period(quarter):
    """YYYYQn -> nombre SEC, ej. 2026Q2 -> '01jun2026-31aug2026'.

    Formato: {01mmmYYYY}-{last_day mmm+2 YYYY}.
    Q4 cruza a febrero del año siguiente.
    """
    year, qn = _parse_quarter(quarter)
    start_month = _QUARTER_START_MONTH[qn]
    end_month = start_month + 2
    end_year = year
    if end_month > 12:
        end_month -= 12
        end_year += 1
    start = "01{}{:04d}".format(_MONTHS[start_month], year)
    end_day = _last_day_of_month(end_year, end_month)
    end = "{:02d}{}{:04d}".format(end_day, _MONTHS[end_month], end_year)
    return start + "-" + end


def latest_published_quarter(today=None):
    """Devuelve el ultimo YYYYQn que probablemente este publicado.

    Regla: SEC publica el dataset ~45-60 dias despues del cierre. Para
    ser conservador, asumimos 60 dias.
    """
    today = today or date.today()
    # Probar Q actual hacia atras. Solo consideramos hasta 60 dias post-cierre.
    candidate_year = today.year
    candidates = []
    for y in (candidate_year, candidate_year - 1):
        for qn in ("Q4", "Q3", "Q2", "Q1"):
            iso = quarter_to_iso_end("{:04d}{}".format(y, qn))
            end = datetime.strptime(iso, "%Y-%m-%d").date()
            if (today - end).days >= 60:
                candidates.append(("{:04d}{}".format(y, qn), end))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0] if candidates else None

def ensure_quarter(quarter, *, force=False,
                   data_dir=None, official_dir=None):
    """Asegura que el trimestre esta ingestado y con official list.

    data_dir: raiz de data/sec_13f (default ROOT/data/sec_13f).
    official_dir: destino de official_list_13f (default data_dir/official_list_13f).

    Devuelve dict con info del proceso.
    """
    year, qn = _parse_quarter(quarter)
    source_period = quarter_to_source_period(quarter)
    base_url = _pick_base_url(source_period)
    data_dir = Path(data_dir) if data_dir else ROOT / "data" / "sec_13f"
    official_dir = Path(official_dir) if official_dir else data_dir / "official_list_13f"

    processed = data_dir / "processed" / quarter
    info = {"quarter": quarter, "source_period": source_period,
            "base_url": base_url, "processed_exists": processed.exists()}

    if processed.exists() and not force:
        print("[SKIP] {} ya ingestado en {}".format(quarter, processed))
    else:
        print("[INGEST] {} desde {}".format(quarter, base_url))
        # ingest_13f gestiona descarga + extract + parse + write + manifest
        result = ingest_13f(
            quarter=quarter,
            source_period=source_period,
            base_dir=str(data_dir),
            force_download=force,
            project_root=str(ROOT),
        )
        info["manifest_path"] = str(result.get("manifest_path", ""))
        info["parquet_paths"] = {k: str(v) for k, v in
                                 result.get("parquet_paths", {}).items()}

    # Official list
    try:
        ol_path = download_official_list(quarter, force=force)
        info["official_list_path"] = str(ol_path)
    except Exception as e:
        print("[WARN] official list no descargada: {}: {}".format(
            type(e).__name__, e))
        info["official_list_error"] = "{}: {}".format(type(e).__name__, e)

    # latest_quarter.txt: solo se escribe si processed/ existe tras la
    # ejecucion. Si ingest fallo, no actualizamos el fichero para no
    # marcar como disponible un trimestre incompleto.
    if processed.exists():
        LATEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        LATEST_FILE.write_text(quarter + "\n", encoding="utf-8", newline="\n")
        info["latest_quarter_file"] = str(LATEST_FILE)
        print("[OK] latest_quarter.txt -> " + quarter)
    else:
        print("[WARN] processed/ no existe; latest_quarter.txt no escrito")

    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarter", help="Trimestre YYYYQn, p.ej. 2026Q2")
    ap.add_argument("--latest", action="store_true",
                    help="Usa el ultimo trimestre probablemente publicado")
    ap.add_argument("--force", action="store_true",
                    help="Ignora cache y redescarga")
    args = ap.parse_args()

    if args.quarter:
        quarter = args.quarter
    elif args.latest:
        quarter = latest_published_quarter()
        if not quarter:
            print("[FAIL] no se pudo determinar el ultimo trimestre")
            return 1
        print("[LATEST] {}".format(quarter))
    else:
        ap.error("--quarter o --latest requerido")

    try:
        info = ensure_quarter(quarter, force=args.force)
    except Exception as e:
        print("[FAIL] {}: {}".format(type(e).__name__, e))
        return 1

    print("\n== Resultado ==")
    for k, v in info.items():
        print("  {}: {}".format(k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())