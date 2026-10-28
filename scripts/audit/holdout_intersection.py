"""Interseccion exacta de los universos holdout (H-03).

Dictamen auditor externo 2026-10-28 (H-03): los parquets reservados
declaran "S&P 500 no en extended" y "Russell 3000 no en extended".
Ambos indices comparten componentes, asi que sus universos pueden
solaparse. No se debe asumir que son disjuntos.

Este script calcula:

- n_tickers de cada uno.
- Interseccion exacta (tickers presentes en ambos).
- Tickers solo en cohort / solo en r3000.
- cohort subset r3000? (si cohort es subconjunto de r3000).

Escribe data/mappings/holdout_intersection.json (versionado).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "data" / "mappings" / "holdout_intersection.json"

PATHS = {
    "cohort": "data/stock_prices_reserved_cohort.parquet",
    "r3000": "data/stock_prices_reserved_r3000.parquet",
}


def _tickers_of(rel_path):
    p = ROOT / rel_path
    if not p.exists():
        raise FileNotFoundError("parquet no existe: " + rel_path)
    df = pd.read_parquet(p)
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError(rel_path + ": columnas no son MultiIndex")
    return sorted(df.columns.get_level_values(1).unique().tolist())


def main(argv=None):
    tickers = {}
    for key, rel in PATHS.items():
        print("[" + key + "] leyendo " + rel + " ...")
        tickers[key] = _tickers_of(rel)
        print("  n_tickers=" + str(len(tickers[key])))

    set_c = set(tickers["cohort"])
    set_r = set(tickers["r3000"])
    inter = sorted(set_c & set_r)
    only_c = sorted(set_c - set_r)
    only_r = sorted(set_r - set_c)
    subset = set_c.issubset(set_r)

    payload = {
        "schema_version": 1,
        "n_cohort": len(set_c),
        "n_r3000": len(set_r),
        "n_intersection": len(inter),
        "n_only_cohort": len(only_c),
        "n_only_r3000": len(only_r),
        "cohort_subset_of_r3000": bool(subset),
        "intersection_sample": inter[:20],
        "only_cohort_sample": only_c[:20],
        "only_r3000_sample": only_r[:20],
        "note": (
            "Si cohort_subset_of_r3000=True, la interseccion es todo "
            "cohort. Los resultados de ambos conjuntos comparten esas "
            "observaciones: no son dos confirmaciones independientes "
            "para los tickers comunes."
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="",
    )
    print("[OK] escrito: " + str(OUT_PATH))
    print("  n_cohort=" + str(len(set_c)) + " n_r3000=" + str(len(set_r)) + " "
          + "interseccion=" + str(len(inter)) + " solo_cohort=" + str(len(only_c)) + " "
          + "solo_r3000=" + str(len(only_r)))
    print("  cohort_subset_of_r3000=" + str(subset))
    return 0


if __name__ == "__main__":
    sys.exit(main())