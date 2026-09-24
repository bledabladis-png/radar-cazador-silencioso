"""Regenera radar_target_catalog.csv desde stock_prices.parquet.

Ciclo: lee los tickers USA del radar, compara con el catalogo actual,
resuelve nuevos via OpenFIGI (TICKER/exchCode=US) y materializa
snapshot + manifest + vista.

Salidas:
  data/mappings/catalog_snapshots/snapshot_YYYYMMDD_NN.csv
  data/mappings/catalog_snapshots/snapshot_YYYYMMDD_NN.sha256
  data/mappings/catalog_manifest.json   (valid_to del activo anterior)
  data/mappings/radar_target_catalog.csv  (vista del snapshot activo)

Sanity checks (aborta sin escribir):
  - nuevos > MAX_NEW (20)
  - desaparecidos > MAX_GONE (10)
  - todos los nuevos con status=MISS

Idempotente: si no hay diff, no escribe nada, exit 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.institutional_accumulation.identity.openfigi_client import map_identifiers
from src.institutional_accumulation.identity.radar_target_catalog import (
    COLUMNS,
    build_from_hits,
    coverage_summary,
    load_radar_tickers,
    write_catalog,
)

STOCK_PRICES = ROOT / "data" / "stock_prices.parquet"
MAPPINGS = ROOT / "data" / "mappings"
SNAPSHOTS = MAPPINGS / "catalog_snapshots"
MANIFEST = MAPPINGS / "catalog_manifest.json"
CATALOG_CSV = MAPPINGS / "radar_target_catalog.csv"

MAX_NEW = 20
MAX_GONE = 10

def _load_current_catalog(path: Path) -> pd.DataFrame:
    """Carga el catalogo actual. Vacio si no existe."""
    if not path.exists():
        return pd.DataFrame(columns=list(COLUMNS))
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _compute_diff(radar_tickers: list, catalog_df: pd.DataFrame) -> dict:
    """Clasifica tickers en nuevos / desaparecidos / estables."""
    current = set(radar_tickers)
    in_catalog = set(catalog_df["radar_ticker"].dropna().astype(str)) if not catalog_df.empty else set()
    return {
        "nuevos": sorted(current - in_catalog),
        "desaparecidos": sorted(in_catalog - current),
        "estables": sorted(current & in_catalog),
    }


def _next_snapshot_id(snapshots_dir: Path, date_tag: str) -> str:
    """Devuelve el siguiente id incremental del dia: YYYYMMDD_NN."""
    if not snapshots_dir.exists():
        return date_tag + "_01"
    existing = [
        p.stem.split(".")[0].rsplit("_", 1)[-1]
        for p in snapshots_dir.glob(f"snapshot_{date_tag}_*.csv")
    ]
    n = max([int(x) for x in existing if x.isdigit()] or [0]) + 1
    return f"{date_tag}_{n:02d}"

def _resolve_new_tickers(tickers: list, *, api_key: str | None) -> dict:
    """Consulta OpenFIGI por TICKER/US y devuelve dict {tk: hit}."""
    if not tickers:
        return {}
    return map_identifiers("TICKER", tickers, exch_code="US", api_key=api_key)


def _build_updated_catalog(
    catalog_df: pd.DataFrame,
    nuevos: list,
    hits: dict,
    *,
    source_date: str,
) -> pd.DataFrame:
    """Concatena el catalogo actual con las filas resueltas de nuevos."""
    if nuevos:
        nuevas_filas = build_from_hits(hits, source_date=source_date)
        nuevas_filas = nuevas_filas[nuevas_filas["radar_ticker"].isin(nuevos)]
        out = pd.concat([catalog_df, nuevas_filas], ignore_index=True)
    else:
        out = catalog_df.copy()
    out = out.drop_duplicates(subset=["radar_ticker"], keep="first")
    return out.sort_values("radar_ticker").reset_index(drop=True)

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _close_active_snapshot(manifest: dict, date_tag: str) -> None:
    """Cierra el snapshot activo (valid_to=null -> date_tag)."""
    for s in manifest.get("snapshots", []):
        if s.get("valid_to") is None:
            s["valid_to"] = date_tag


def _load_manifest() -> dict:
    if not MANIFEST.exists():
        return {"schema_version": 1, "snapshots": []}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-date", default=None,
                    help="YYYY-MM-DD (default: hoy UTC). Overridable para tests.")
    ap.add_argument("--dry-run", action="store_true",
                    help="No escribe ficheros, solo imprime el diff.")
    args = ap.parse_args()

    date_tag = args.source_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print("=" * 72)
    print("Regenerar radar_target_catalog  (source_date=" + date_tag + ")")
    print("=" * 72)

    radar = load_radar_tickers(STOCK_PRICES)
    print("\ntickers en stock_prices: " + str(len(radar)))

    cat_actual = _load_current_catalog(CATALOG_CSV)
    print("tickers en catalogo:     " + str(len(cat_actual)))

    diff = _compute_diff(radar, cat_actual)
    print("\n--- diff ---")
    print("  nuevos:        " + str(len(diff["nuevos"])) + "  " + str(diff["nuevos"][:10]))
    print("  desaparecidos: " + str(len(diff["desaparecidos"])) + "  " + str(diff["desaparecidos"][:10]))
    print("  estables:      " + str(len(diff["estables"])))

    if len(diff["nuevos"]) > MAX_NEW:
        print(f"\n[ABORT] nuevos={len(diff['nuevos'])} > MAX_NEW={MAX_NEW}")
        return 1
    if len(diff["desaparecidos"]) > MAX_GONE:
        print(f"\n[ABORT] desaparecidos={len(diff['desaparecidos'])} > MAX_GONE={MAX_GONE}")
        return 1
    if not diff["nuevos"]:
        print("\n[OK] sin cambios. No se escribe nada.")
        return 0

    api_key = os.environ.get("OPENFIGI_API_KEY", "").strip() or None
    print("\nAPI key presente: " + str(bool(api_key)))
    print("Resolviendo " + str(len(diff["nuevos"])) + " tickers nuevos...")
    hits = _resolve_new_tickers(diff["nuevos"], api_key=api_key)

    n_ok = sum(1 for h in hits.values() if h.get("ok"))
    print("  con hit: " + str(n_ok) + "/" + str(len(diff["nuevos"])))
    if diff["nuevos"] and n_ok == 0:
        print("\n[ABORT] ningun ticker nuevo resuelto. OpenFIGI caido o sin hits.")
        return 1

    nuevo_catalogo = _build_updated_catalog(
        cat_actual, diff["nuevos"], hits, source_date=date_tag,
    )
    print("\ncoverage: " + str(coverage_summary(nuevo_catalogo)))

    if args.dry_run:
        print("\n[DRY-RUN] nada escrito.")
        return 0

    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    version_id = _next_snapshot_id(SNAPSHOTS, date_tag)
    snap_csv = SNAPSHOTS / f"snapshot_{version_id}.csv"
    write_catalog(nuevo_catalogo, snap_csv)
    sha = _sha256(snap_csv)
    (SNAPSHOTS / f"snapshot_{version_id}.sha256").write_text(sha + "\n", encoding="utf-8")

    manifest = _load_manifest()
    _close_active_snapshot(manifest, date_tag)
    manifest["snapshots"].append({
        "version_id": version_id,
        "valid_from": date_tag,
        "valid_to": None,
        "sha256": sha,
        "csv_path": f"catalog_snapshots/snapshot_{version_id}.csv",
        "rows": int(len(nuevo_catalogo)),
        "producer": "regenerate_radar_catalog.py",
    })
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    write_catalog(nuevo_catalogo, CATALOG_CSV)

    print("\n[OK] snapshot=" + version_id + "  sha256=" + sha[:16] + "...")
    print("[OK] manifest y vista actualizados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())