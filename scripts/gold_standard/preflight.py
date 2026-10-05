# -*- coding: utf-8 -*-
"""Preflight integration test sobre datos reales.

Dictamen P1.6: antes de generar paquetes para anotadores, verificar
la cadena completa sobre stock_prices.parquet.

NO genera PNGs. NO escribe a outputs/gold_standard/capa1.
Solo verifica y reporta. Salida: outputs/audit/preflight_*.json.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# Asegurar que ROOT esta en sys.path antes de importar paquetes del proyecto.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from scripts.gold_standard.constants import (
    MIN_GAP_A_SESSIONS,
    N_NEG_A,
    N_POS_A,
    SEED_GLOBAL,
)
from scripts.gold_standard.sampling_a import build_metadata, sample_a
from scripts.gold_standard.sampling_b import (
    build_estratos,
    collapse_estratos,
    sample_b,
)
from scripts.gold_standard.sampling_frame import (
    build_sampling_frame,
    load_dataset,
)
from scripts.gold_standard.sector_map import (
    coverage_report,
    load_sector_map,
    sector_map_dict,
)

ROOT = _ROOT
OUT_DIR = ROOT / "outputs" / "audit"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_preflight(
    n_B_test: int = 200,
    seed: int = SEED_GLOBAL,
    verbose: bool = True,
) -> dict:
    """Ejecuta la cadena completa en modo preflight."""
    t0 = time.time()
    report = {
        "modo": "preflight",
        "seed": seed,
        "n_B_test": n_B_test,
        "pasos": {},
    }

    # 1. Dataset
    if verbose:
        print("[preflight] cargando dataset...")
    dataset = load_dataset()
    tickers_dataset = set(dataset.columns.get_level_values(1).unique())
    report["pasos"]["dataset"] = {
        "shape": list(dataset.shape),
        "n_tickers": len(tickers_dataset),
    }

    # 2. Sector map
    if verbose:
        print("[preflight] cargando sector_map...")
    sector_df = load_sector_map(dataset_tickers=tickers_dataset)
    smap = sector_map_dict(sector_df)
    rep = coverage_report(sector_df, tickers_dataset)
    report["pasos"]["sector_map"] = {
        "n_con_sector": rep["n_con_sector"],
        "n_sin_sector": rep["n_sin_sector"],
        "n_dataset": rep["n_dataset"],
        "por_sector": rep["por_sector"],
    }

    # 3. Sampling frame
    if verbose:
        print("[preflight] construyendo sampling frame...")
    sf = build_sampling_frame(dataset)
    report["pasos"]["sampling_frame"] = {
        "n_frame": sf.n_frame,
        "n_tickers_with_frame": sf.diagnostics["n_tickers_with_frame"],
        "filtro_sectorial": sf.diagnostics["filtro_sectorial"],
    }

    frame = sf.episodes.copy()

    # 4. Metadata con sector UNKNOWN
    if verbose:
        print("[preflight] metadata...")
    meta = build_metadata(frame, smap)
    n_unknown = int((meta["sector"] == "UNKNOWN").sum())
    n_unknown_tickers = int(
        meta[meta["sector"] == "UNKNOWN"]["ticker"].nunique()
    )
    report["pasos"]["metadata"] = {
        "n_episodios": len(meta),
        "n_unknown_episodios": n_unknown,
        "n_unknown_tickers": n_unknown_tickers,
        "sectores_unicos": sorted(meta["sector"].unique().tolist()),
    }

    # 5. Detector flags: ejecucion REAL del detector sobre el frame.
    if verbose:
        print("[preflight] calculando detect_sow real (puede tardar ~35s)...")
    from scripts.gold_standard.run_capa1 import compute_detector_flags
    t_det0 = time.time()
    flags = compute_detector_flags(frame, dataset)
    n_pos = int(flags.sum())
    n_neg = int(len(flags) - n_pos)
    report["pasos"]["detector_flags"] = {
        "n_total": int(len(flags)),
        "n_positivos": n_pos,
        "n_negativos": n_neg,
        "tiempo_seg": round(time.time() - t_det0, 2),
    }
    if verbose:
        print(f"[preflight] detect_sow: {n_pos} positivos / {len(flags)}")

    # 6. Muestreo B
    if verbose:
        print("[preflight] muestreo B...")
    estratos_ini = build_estratos(meta)
    estratos_col = collapse_estratos(estratos_ini)
    n_sectores_orig = int(estratos_ini["sector"].nunique())
    n_sectores_col = int(estratos_col["sector"].nunique())
    report["pasos"]["estratos"] = {
        "n_estratos_iniciales": len(estratos_ini),
        "n_estratos_colapsados": len(estratos_col),
        "n_sectores_orig": n_sectores_orig,
        "n_sectores_col": n_sectores_col,
        "sectores_finales": sorted(estratos_col["sector"].unique().tolist()),
    }
    sB = sample_b(meta, n_B=n_B_test, estratos_finales=estratos_col, seed=seed)
    report["pasos"]["muestra_B"] = {
        "n": len(sB),
        "n_tickers_unicos": int(sB["ticker"].nunique()),
        "pesos_min": float(sB["w_h"].min()),
        "pesos_max": float(sB["w_h"].max()),
    }

    # 7. Muestreo A excluyendo B
    if verbose:
        print("[preflight] muestreo A excluye B...")
    # Muestra A en modo no estricto: usa lo que haya, no falla.
    # El preflight reporta cuantos positivos hay realmente.
    sA = sample_a(
        meta, flags,
        n_pos=N_POS_A, n_neg=N_NEG_A,
        min_gap_sessions=MIN_GAP_A_SESSIONS,
        seed=seed,
        excluir=sB[["ticker", "t"]],
        strict=False,
    )
    n_pos_final = int((sA["detect_sow"] == 1).sum()) if len(sA) > 0 else 0
    n_neg_final = int((sA["detect_sow"] == 0).sum()) if len(sA) > 0 else 0
    report["pasos"]["muestra_A"] = {
        "n": len(sA),
        "n_tickers_unicos": int(sA["ticker"].nunique()) if len(sA) > 0 else 0,
        "n_positivos_obtenidos": n_pos_final,
        "n_negativos_obtenidos": n_neg_final,
        "n_pos_objetivo": N_POS_A,
        "n_neg_objetivo": N_NEG_A,
        "nota": (
            "modo preflight (no estricto): si detect_sow dispara poco "
            "en el universo, Muestra A sera menor de 200+200."
        ),
    }

    # 8. Verificacion A ∩ B = ∅
    key_A = set(zip(sA["ticker"], sA["t"]))
    key_B = set(zip(sB["ticker"], sB["t"]))
    inter = key_A & key_B
    report["pasos"]["disjuncion"] = {
        "n_intersect": len(inter),
        "ok": len(inter) == 0,
    }

    # 9. Blind IDs
    if verbose:
        print("[preflight] blind IDs...")
    from scripts.gold_standard.annotation import build_blind_muestra
    comunes = ["ticker", "t", "muestra", "sector", "periodo"]
    sA_ = sA.copy()
    sB_ = sB.copy()
    for c in comunes:
        if c not in sB_.columns:
            sB_[c] = "B"
    sAB = pd.concat([sA_[comunes], sB_[comunes]], ignore_index=True)
    blind = build_blind_muestra(sAB, seed=seed)
    report["pasos"]["blind_ids"] = {
        "n": len(blind.mapping),
        "unicos": bool(blind.mapping["blind_id"].is_unique),
        "min": int(blind.mapping["blind_id"].min()),
        "max": int(blind.mapping["blind_id"].max()),
    }

    # Resumen final
    report["duracion_seg"] = round(time.time() - t0, 2)
    report["checks_ok"] = {
        "frame_no_filtra_sector": not sf.diagnostics["filtro_sectorial"],
        "A_B_disjuntas": len(inter) == 0,
        "blind_ids_unicos": bool(blind.mapping["blind_id"].is_unique),
        "n_estratos_colapsados_le_iniciales": (
            len(estratos_col) <= len(estratos_ini)
        ),
    }
    report["checks_ok"]["todos_pasan"] = all(report["checks_ok"].values())
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description="Preflight Gold Standard")
    p.add_argument("--n-B", type=int, default=200)
    p.add_argument("--seed", type=int, default=SEED_GLOBAL)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    report = run_preflight(n_B_test=args.n_B, seed=args.seed)

    out = args.out
    if out is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = OUT_DIR / f"preflight_gold_standard_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[preflight] reporte escrito: {out}")
    print(f"[preflight] checks_ok.todos_pasan = {report['checks_ok']['todos_pasan']}")
    return 0 if report["checks_ok"]["todos_pasan"] else 1


if __name__ == "__main__":
    sys.exit(main())