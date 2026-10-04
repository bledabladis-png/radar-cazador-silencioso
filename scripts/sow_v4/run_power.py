"""Ejecuta el power analysis de la seccion 12.

Uso:
    py scripts/sow_v4/run_power.py --smoke
    py scripts/sow_v4/run_power.py --n-replicas 2000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.sow_v4 import config, power, regime
from scripts.sow_v4.episodes import precompute_base

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 replica, 1 escenario")
    ap.add_argument("--n-replicas", type=int, default=config.POWER_N_REPLICAS)
    ap.add_argument("--scenarios", type=str, default="", help="csv nombres, vacio=todos")
    args = ap.parse_args()

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Cargando dataset...")
    df = pd.read_parquet(config.DATA_PARQUET)
    close = df.xs("Close", axis=1, level=0)
    cov = close.notna().sum()
    tickers = sorted(cov[cov >= config.MIN_OBS].index.tolist())
    print(f"      tickers={len(tickers)}")

    print("[2/5] Precompute feats + starts_cache...")
    import scripts.calibrate_wyckoff_sow_5b4bis as calib
    calib.DATA_PARQUET = config.DATA_PARQUET
    feats, starts_cache = precompute_base(df, tickers)
    print(f"      feats={len(feats)}")

    print("[3/5] Regimen de mercado...")
    sector_map = _load_sector_map()
    sector_returns = regime.compute_sector_returns(df, sector_map)
    print(f"      sector_returns shape={sector_returns.shape}")

    print("[4/5] Precompute combos (240)...")
    combo_eps = power.precompute_combos(feats, starts_cache, sector_returns)
    print(f"      combos con episodios={len(combo_eps)}")

    print("[5/5] Ejecutando escenarios...")
    scenarios = config.POWER_SCENARIOS
    if args.smoke:
        scenarios = scenarios[:1]
        n_rep = 1
    elif args.scenarios:
        wanted = set(args.scenarios.split(","))
        scenarios = [s for s in scenarios if s[0] in wanted]
        n_rep = args.n_replicas
    else:
        n_rep = args.n_replicas

    results = []
    for name, dn, ds in scenarios:
        t0 = time.time()
        print(f"      {name}: RD_normal={dn} RD_stress={ds} R={n_rep}")
        res = power.run_scenario(
            name, dn, ds, combo_eps, feats, starts_cache, sector_returns, n_rep,
        )
        res["elapsed_s"] = round(time.time() - t0, 1)
        results.append(res)
        print(f"        -> {res['counts']} ({res['elapsed_s']}s)")

    out = config.OUT_DIR / "power_analysis.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[OK] escrito {out}")


def _load_sector_map() -> dict:
    """Placeholder: cargar mapa ticker->sector. En el proyecto real viene
    de config.settings o de un fichero de mapping."""
    from config import settings
    if hasattr(settings, "TICKER_TO_SECTOR"):
        return dict(settings.TICKER_TO_SECTOR)
    return {}


if __name__ == "__main__":
    main()