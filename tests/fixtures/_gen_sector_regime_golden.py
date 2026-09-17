"""DT1 Fase 0 (2026-09-17): generador del golden de caracterizacion.

NO es codigo productivo. Se ejecuta una vez para crear
tests/fixtures/sector_regime_golden.json.

Se conserva en tests/fixtures/_gen_sector_regime_golden.py para
documentar como se genero el golden (patron DT2).

Contrato congelado: ranking, regime, top3.
Diagnostico congelado: last_scores.
NO se congela 'components' (se elimina en DT1 Fase 1).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from config.tickers import MARKET_TICKERS
from regimes.sector_regime import compute_sector_scores


SEED = 42
N_DAYS = 300


def make_synthetic_df(seed=SEED, n=N_DAYS):
    sectors = MARKET_TICKERS["sectors"]
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.RandomState(seed)
    data = {}
    data[("Close", "^GSPC")] = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    for k, s in enumerate(sectors):
        drift = 0.0001 * (k + 1)
        close = 100 * np.cumprod(1 + rng.normal(drift, 0.015, n))
        data[("Close", s)] = close
        data[("Open", s)] = close * 0.998
        data[("High", s)] = close * 1.01
        data[("Low", s)] = close * 0.99
        data[("Volume", s)] = rng.randint(1000000, 5000000, n)
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def main():
    df = make_synthetic_df()
    res = compute_sector_scores(df)
    if res is None:
        raise SystemExit("compute_sector_scores devolvio None")

    ranking_ser = [list(r) for r in res["ranking"]]
    last_scores_dict = {k: float(v) for k, v in res["last_scores"].items()}
    top3 = [str(s) for s in res["last_scores"].head(3).index]

    golden = {
        "reference_date": "2026-09-17",
        "generator": "tests/fixtures/_gen_sector_regime_golden.py",
        "seed": SEED,
        "n_days": N_DAYS,
        "synthetic_df": "RandomState(seed), fechas 2024-01-01 freq=B",
        "contract": {
            "ranking": ranking_ser,
            "regime": res["regime"],
            "top3": top3,
        },
        "diagnostic": {
            "last_scores": last_scores_dict,
        },
    }
    out = Path("tests/fixtures/sector_regime_golden.json")
    out.write_bytes(
        json.dumps(golden, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    )
    print(f"OK -> {out}")
    print(f"regime = {res['regime']}")
    print(f"top3   = {top3}")
    print()
    print("ranking:")
    for t, n, s, w in res["ranking"]:
        print(f"  {t:5s} {n:30s} {s:+.4f}  {w}")


if __name__ == "__main__":
    main()