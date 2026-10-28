"""Manifiesto inmutable de los parquets holdout (SOW-HOLDOUT-IMMUTABLE H-01).

Dictamen auditor externo 2026-10-28 (H-01): los parquets reservados
estan excluidos de Git. El contrato reconoce que deben regenerarse si
se pierden, pero no fija huella del contenido aprobado ni regla
objetiva de equivalencia. Si el proveedor revisa historicos (splits,
dividendos, tickers), el generador podria reconstruir un parquet
distinto con el mismo rango temporal.

Este script cierra H-01:

- SHA-256 del fichero entero. Detecta CUALQUIER cambio de bytes.
- Huella canonica del esqueleto: date_min, date_max, n_rows, n_tickers,
  fields ordenados, tickers ordenados.
- tickers_sha256: hash solo de la lista de tickers. Para comparar
  universos entre parquets.
- Manifest versionado en data/mappings/holdout_manifest.json. Los
  parquets NO se versionan; el manifest SI.

Modos:
  py scripts/audit/holdout_manifest.py            # genera si no existe; WARN si difiere (no escribe)
  py scripts/audit/holdout_manifest.py --verify   # compara; exit 1 si difiere (no escribe)
  py scripts/audit/holdout_manifest.py --update   # regenera (acto explicito)

Contrato de no-sobrescritura silenciosa: ni el modo default ni
--verify escriben cuando el manifest existe y difiere. Solo --update
escribe. Regenerar un holdout ya sellado debe ser un acto explicito.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MANIFEST_PATH = ROOT / "data" / "mappings" / "holdout_manifest.json"

HOLDOUTS = {
    "cohort": {
        "path": "data/stock_prices_reserved_cohort.parquet",
        "universe_name": "S&P 500 no en stock_prices_extended.parquet",
        "generator": "scripts/fetch_reserved_cohort.py",
        "source": "yahoo",
    },
    "r3000": {
        "path": "data/stock_prices_reserved_r3000.parquet",
        "universe_name": "Russell 3000 no en stock_prices_extended.parquet",
        "generator": "scripts/fetch_reserved_cohort_r3000.py",
        "source": "yahoo",
    },
}


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_meta(df):
    if isinstance(df.columns, pd.MultiIndex):
        fields = sorted(df.columns.get_level_values(0).unique().tolist())
        tickers = sorted(df.columns.get_level_values(1).unique().tolist())
    else:
        fields = list(df.columns)
        tickers = []
    return {
        "index_name": str(df.index.name) if df.index.name else None,
        "date_min": str(pd.Timestamp(df.index.min()).date()),
        "date_max": str(pd.Timestamp(df.index.max()).date()),
        "n_rows": int(len(df)),
        "n_tickers": int(len(tickers)),
        "n_fields": int(len(fields)),
        "fields": fields,
        "tickers": tickers,
    }


def _canonical_sha256(meta):
    payload = {
        "date_min": meta["date_min"],
        "date_max": meta["date_max"],
        "n_rows": meta["n_rows"],
        "n_tickers": meta["n_tickers"],
        "n_fields": meta["n_fields"],
        "fields": sorted(meta["fields"]),
        "tickers": sorted(meta["tickers"]),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _tickers_sha256(tickers):
    payload = "\n".join(sorted(tickers)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_entry(rel_path, universe_name, generator, source):
    p = ROOT / rel_path
    if not p.exists():
        raise FileNotFoundError("parquet no existe: " + rel_path)
    df = pd.read_parquet(p)
    meta = _extract_meta(df)
    return {
        "path": rel_path,
        "universe_name": universe_name,
        "generator": generator,
        "source": source,
        "sha256": _sha256_file(p),
        "bytes": int(p.stat().st_size),
        "content_sha256": _canonical_sha256(meta),
        "tickers_sha256": _tickers_sha256(meta["tickers"]),
        "index_name": meta["index_name"],
        "date_min": meta["date_min"],
        "date_max": meta["date_max"],
        "n_rows": meta["n_rows"],
        "n_tickers": meta["n_tickers"],
        "n_fields": meta["n_fields"],
        "fields": meta["fields"],
    }


def build_manifest():
    holdouts = {}
    for key, spec in HOLDOUTS.items():
        print("[" + key + "] hasheando " + spec["path"] + " ...")
        holdouts[key] = _build_entry(
            spec["path"], spec["universe_name"],
            spec["generator"], spec["source"],
        )
        print("  sha256=" + holdouts[key]["sha256"][:16] + "... "
              + "n_tickers=" + str(holdouts[key]["n_tickers"]) + " "
              + "range=" + holdouts[key]["date_min"] + ".." + holdouts[key]["date_max"])
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "holdouts": holdouts,
    }


def compare(existing, current):
    errores = []
    ex = existing.get("holdouts") or {}
    cu = current.get("holdouts") or {}
    for key in sorted(set(ex) | set(cu)):
        if key not in ex:
            errores.append(key + ": aparece en current, no en manifest")
            continue
        if key not in cu:
            errores.append(key + ": desaparece del current")
            continue
        for campo in (
            "sha256", "bytes", "content_sha256", "tickers_sha256",
            "date_min", "date_max", "n_rows", "n_tickers", "n_fields",
        ):
            old = ex[key].get(campo)
            new = cu[key].get(campo)
            if old != new:
                errores.append(key + "." + campo + ": " + repr(old) + " -> " + repr(new))
    return errores


def main(argv=None):
    ap = argparse.ArgumentParser(description="Manifiesto inmutable de holdouts SOW")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--verify", action="store_true",
                   help="compara el manifest existente. Exit 1 si difiere. No escribe.")
    g.add_argument("--update", action="store_true",
                   help="regenera el manifest (acto explicito).")
    args = ap.parse_args(argv)

    try:
        current = build_manifest()
    except FileNotFoundError as e:
        print("[ERROR] " + str(e))
        return 2

    existing = None
    if MANIFEST_PATH.exists():
        try:
            existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print("[ERROR] manifest ilegible: " + str(e))
            return 2

    if existing is None:
        if args.verify:
            print("[FAIL] no hay manifest. Ejecutar sin --verify para generarlo.")
            return 1
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(
            json.dumps(current, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8", newline="",
        )
        print("[OK] manifest creado: " + str(MANIFEST_PATH))
        return 0

    errores = compare(existing, current)
    if not errores:
        print("[OK] manifest coincide con los parquets actuales")
        return 0

    print("[DIFF] " + str(len(errores)) + " diferencia(s) entre manifest y parquets:")
    for e in errores:
        print("  " + e)

    if args.update:
        MANIFEST_PATH.write_text(
            json.dumps(current, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8", newline="",
        )
        print("[OK] manifest ACTUALIZADO: " + str(MANIFEST_PATH))
        return 0
    if args.verify:
        print("[FAIL] --verify: los parquets no coinciden con el manifest sellado.")
        return 1
    print("[WARN] manifest NO modificado. Usar --update si el cambio es deliberado.")
    return 1


if __name__ == "__main__":
    sys.exit(main())