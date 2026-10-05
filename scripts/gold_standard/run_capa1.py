# -*- coding: utf-8 -*-
"""Orquestador Capa 1 (Gold Standard del detector SOW).

Flujo:
  1. Cargar stock_prices.parquet
  2. Cargar sector_map (top-20 por ETF)
  3. Construir sampling_frame
  4. Calcular flags detect_sow en cada (ticker, t) del frame
  5. Muestreo A (enriquecida, min_gap=120)
  6. Muestreo B (representativa, estratos colapsados)
  7. Dimensionar n_B por simulacion
  8. Asignar blind_ids
  9. Renderizar PNGs
 10. Escribir CSVs de anotacion
 11. Manifest con SHA-256 de todos los artefactos

Modo por defecto: --dry-run. NO toca disco ni calcula el detector.
Para ejecutar de verdad: --go. Solo autorizado tras firma del auditor.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

from scripts.gold_standard.annotation import (
    build_blind_muestra,
    write_all_csvs,
)
from scripts.gold_standard.constants import (
    CAPACIDAD_MAX_NB,
    MIN_GAP_A_SESSIONS,
    N_NEG_A,
    N_POS_A,
    SEED_GLOBAL,
)
from scripts.gold_standard.render import render_all
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

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR_DEFAULT = ROOT / "outputs" / "gold_standard" / "capa1"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Orquestador Capa 1")
    p.add_argument("--go", action="store_true",
                   help="Ejecucion real. Sin esto, dry-run.")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--n-B", type=int, default=None,
                   help="Tamaño Muestra B. Si None, se dimensiona.")
    p.add_argument("--skip-power", action="store_true",
                   help="Saltar dimensionamiento de n_B.")
    p.add_argument("--seed", type=int, default=SEED_GLOBAL)
    return p.parse_args(argv)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_detector_flags(
    frame_episodes: pd.DataFrame,
    dataset: pd.DataFrame,
) -> pd.Series:
    """Calcula detect_sow en cada (ticker, t) del frame.

    Llama al detector con parametros congelados (FROZEN_V19_PARAMS).
    Index: MultiIndex (ticker, t). Valor: int 0/1.
    """
    from scripts.gold_standard.constants import FROZEN_V19_PARAMS
    from indicators.wyckoff_v1 import detect_sow

    flags = pd.Series(
        0, index=pd.MultiIndex.from_arrays(
            [frame_episodes["ticker"], frame_episodes["t"]],
            names=["ticker", "t"],
        ),
        dtype=int,
    )

    tickers_unicos = frame_episodes["ticker"].unique()
    for tk in tickers_unicos:
        sub = frame_episodes[frame_episodes["ticker"] == tk]
        try:
            df_tk = dataset.xs(tk, axis=1, level=1)
        except KeyError:
            continue
        if len(df_tk) < 200:
            continue
        try:
            sow_series = detect_sow(
                df_tk, tk,
                window=FROZEN_V19_PARAMS["window"],
                x_atr=FROZEN_V19_PARAMS["x_atr"],
                y_vol=FROZEN_V19_PARAMS["y_vol"],
            )
        except (KeyError, ValueError, TypeError, IndexError):
            continue
        if sow_series is None or len(sow_series) == 0:
            continue
        for t in sub["t"]:
            if t in sow_series.index:
                flags.loc[(tk, t)] = int(sow_series.loc[t])
    return flags

def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()

    print(f"[capa1] modo: {'GO (ejecucion real)' if args.go else 'DRY-RUN'}")
    print(f"[capa1] out_dir: {args.out_dir}")
    print(f"[capa1] seed: {args.seed}")

    # 1. Dataset
    print("[capa1] cargando dataset...")
    dataset = load_dataset()
    print(f"[capa1] dataset shape: {dataset.shape}")

    # 2. Sector map
    print("[capa1] cargando sector map...")
    dataset_tickers = set(dataset.columns.get_level_values(1).unique())
    sector_df = load_sector_map(dataset_tickers=dataset_tickers)
    smap = sector_map_dict(sector_df)
    rep = coverage_report(sector_df, dataset_tickers)
    print(f"[capa1] tickers con sector: {rep['n_con_sector']}"
          f" / {rep['n_dataset']}")
    print(f"[capa1] distribucion por sector: {rep['por_sector']}")

    # 3. Sampling frame
    print("[capa1] construyendo sampling frame...")
    sf = build_sampling_frame(dataset)
    print(f"[capa1] sampling frame: {sf.n_frame} episodios"
          f" sobre {sf.diagnostics['n_tickers_with_frame']} tickers")

    # Universo completo. NO filtrar por sector (dictamen DOC 53).
    # Tickers sin sector reciben UNKNOWN en build_metadata.
    frame = sf.episodes.copy()
    print(f"[capa1] frame universo completo: {len(frame)}")

    if not args.go:
        print("[capa1] DRY-RUN. No se calcula detector ni se escribe nada.")
        print(f"[capa1] duracion: {time.time() - t0:.2f}s")
        return 0

    # 4. Detector flags sobre universo completo
    print("[capa1] calculando detect_sow sobre el frame...")
    flags = compute_detector_flags(frame, dataset)
    print(f"[capa1] positivos detect_sow: {int(flags.sum())}"
          f" / {len(flags)}")

    # 5. Metadata (sector UNKNOWN si no hay mapping)
    meta = build_metadata(frame, smap)
    n_unknown = int((meta["sector"] == "UNKNOWN").sum())
    print(f"[capa1] metadata: {meta.shape}, UNKNOWN: {n_unknown}")

    # 6. Dimensionar n_B (o usar el del CLI)
    if args.skip_power or args.n_B is not None:
        n_B = args.n_B if args.n_B is not None else 300
        print(f"[capa1] n_B fijado por CLI: {n_B}")
    else:
        from scripts.gold_standard.power import dimensionar_n_B
        print("[capa1] dimensionando n_B por simulacion...")
        estratos = build_estratos(meta)
        estratos_col = collapse_estratos(estratos)
        res = dimensionar_n_B(estratos_col, n_min=100, n_max=CAPACIDAD_MAX_NB,
                              paso=50, B=100, seed=args.seed)
        if res["excede_capacidad"]:
            print(f"[capa1] ABORTAR: n_B supera {CAPACIDAD_MAX_NB}")
            return 1
        n_B = res["n_B_min"]
        print(f"[capa1] n_B minimo: {n_B}")

    # 7. Muestreo B PRIMERO (dictamen P0.3: A y B disjuntas)
    print("[capa1] muestreo B (representativa)...")
    estratos = build_estratos(meta)
    estratos_col = collapse_estratos(estratos)
    sB = sample_b(meta, n_B=n_B, estratos_finales=estratos_col, seed=args.seed)
    print(f"[capa1] muestra B: {len(sB)}")

    # 8. Muestreo A sobre frame residual (excluye B)
    print("[capa1] muestreo A (enriquecida, excluye B)...")
    sA = sample_a(meta, flags, n_pos=N_POS_A, n_neg=N_NEG_A,
                  min_gap_sessions=MIN_GAP_A_SESSIONS, seed=args.seed,
                  excluir=sB[["ticker", "t"]])
    print(f"[capa1] muestra A: {len(sA)}")

    # Verificar A ∩ B = ∅
    key_A = set(zip(sA["ticker"], sA["t"]))
    key_B = set(zip(sB["ticker"], sB["t"]))
    inter = key_A & key_B
    if inter:
        print(f"[capa1] ABORTAR: A y B comparten {len(inter)} casos")
        return 1
    print("[capa1] A y B disjuntas: OK")

    return _finalize(args, dataset, smap, sA, sB, meta, n_B, t0)

def _finalize(args, dataset, smap, sA, sB, meta, n_B, t0):
    """Blind IDs, render, CSVs, manifest."""
    # 9. Concatenar (A y B ya disjuntas por construccion)
    sA_ = sA.copy()
    sB_ = sB.copy()
    # Columnas comunes minimas
    for c in ("sector", "periodo"):
        if c not in sB_.columns and c in sA_.columns:
            sB_[c] = sB_["estrato_sector"] if c == "sector" else sB_["estrato_periodo"]
    comunes = ["ticker", "t", "muestra", "sector", "periodo"]
    sAB = pd.concat([sA_[comunes], sB_[comunes]], ignore_index=True)
    print(f"[capa1] muestra conjunta: {len(sAB)}")

    # 10. Blind IDs
    print("[capa1] asignando blind_ids...")
    blind = build_blind_muestra(sAB, seed=args.seed)

    # 11. Directorio y CSV
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = write_all_csvs(blind, args.out_dir)
    print(f"[capa1] CSVs escritos: {list(csv_paths.keys())}")

    # 12. Render PNGs
    print("[capa1] renderizando PNGs...")
    cases_dir = args.out_dir / "cases"
    render_res = render_all(blind.mapping, dataset, cases_dir)
    print(f"[capa1] PNGs: ok={render_res['ok']} fail={render_res['fail']}")

    # 13. Manifest
    manifest = {
        "estudio": "gold_standard_capa1",
        "seed": args.seed,
        "n_muestra_A": int(len(sA)),
        "n_muestra_B": int(len(sB)),
        "n_conjunta": int(len(sAB)),
        "n_pngs_ok": render_res["ok"],
        "n_pngs_fail": render_res["fail"],
        "duracion_segundos": round(time.time() - t0, 2),
        "csv_sha256": {k: sha256_file(v) for k, v in csv_paths.items()},
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[capa1] manifest: {manifest_path}")
    print(f"[capa1] duracion total: {manifest['duracion_segundos']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())