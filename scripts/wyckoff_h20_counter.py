# -*- coding: utf-8 -*-
# Calcula n_confirmed_H20_complete y lo persiste en outputs/state/.
#
# Soporte para WYCKOFF-5BX-INSTRUMENTACION. Lee los 6 CSVs wyckoff_*
# de outputs/history/ si existen, concatena, y llama a
# count_confirmed_h20_complete. Escribe outputs/state/wyckoff_5bx_counter.json.
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from indicators.wyckoff_h20 import count_confirmed_h20_complete

HISTORY_DIR = REPO_ROOT / "outputs" / "history"
STATE_DIR = REPO_ROOT / "outputs" / "state"
OUTPUT_JSON = STATE_DIR / "wyckoff_5bx_counter.json"

CSV_NAMES = [
    "wyckoff_usa_sector_accumulation.csv",
    "wyckoff_usa_sector_distribution.csv",
    "wyckoff_usa_index_accumulation.csv",
    "wyckoff_usa_index_distribution.csv",
    "wyckoff_eu_index_accumulation.csv",
    "wyckoff_eu_index_distribution.csv",
]


def _load_all():
    frames = []
    for name in CSV_NAMES:
        p = HISTORY_DIR / name
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except (OSError, pd.errors.ParserError, ValueError):
            continue
        if df is None or df.empty:
            continue
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def main():
    df = _load_all()
    result = count_confirmed_h20_complete(df)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("[OK] " + str(OUTPUT_JSON))
    print("  n_confirmed_h20_complete = " + str(result["n_confirmed_h20_complete"]))
    print("  n_confirmed_total        = " + str(result["n_confirmed_total"]))
    print("  threshold_5bx            = " + str(result["threshold_5bx"]))
    print("  threshold_reached        = " + str(result["threshold_reached"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
