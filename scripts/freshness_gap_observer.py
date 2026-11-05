"""Observatorio de gap de frescura (TEST-FRESHNESS-UMBRAL-VS-CADENCIA).

Ejecutado por daily_run.yml post-run.py, ANTES del step
"Validate freshness post-run". Append-only a
outputs/history/freshness_gap_observation.csv. No bloquea: exit 0
siempre (continue-on-error: true en el YAML).

Motivo: los 3 tests de tests/test_freshness.py fallan en local con
parquets del bot (no se auto-resuelven con el refresh). Hipotesis:
los umbrales (MAX_DAILY_AGE=4, MAX_DAILY_AGE_EU=5, MAX_DQ_AGE=3) son
demasiado estrictos para la cadencia real del bot (5 slots/dia UTC:
23:17, 03:17, 05:17, 07:17, 11:17). Este script registra el gap real
para decidir con datos, no con intuicion.

Criterio de decision (2026-11-10):
- >=3 de 5 dias con gap > MAX_DAILY_AGE: umbral demasiado estricto,
  ajustar (con justificacion: tolerar festivo + latencia real).
- <2 de 5 dias con gap: el gap es sintoma de otra cosa (latencia
  Yahoo/FRED, no umbral). Investigar por otro lado.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MANIFEST_PATH = ROOT / "data" / "market_data.parquet.manifest.json"
OUT_CSV = ROOT / "outputs" / "history" / "freshness_gap_observation.csv"
COLUMNS = [
    "target_session",
    "expected_session",
    "effective_session",
    "last_date",
    "gap_days_calendar",
    "gap_days_business",
    "coverage_pct_effective",
    "n_tickers_missing",
    "manifest_status",
    "test_would_fail",
]

# Umbrales copiados de tests/test_freshness.py. Si cambian alli, este
# observatorio debe actualizarse (test centinela lo verifica).
MAX_DAILY_AGE = 4


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print("[WARN] manifest invalido: " + str(e))
        return {}


def _business_days_between(d1: date, d2: date) -> int:
    if d1 == d2:
        return 0
    lo, hi = (d1, d2) if d1 < d2 else (d2, d1)
    return len(pd.bdate_range(lo, hi)) - 1


def _row_from_manifest(m: dict) -> dict | None:
    q = m.get("quality") or {}
    last_date = q.get("last_date")
    expected = q.get("expected_session")
    if not last_date or not expected:
        return None
    try:
        d_last = date.fromisoformat(str(last_date))
        d_exp = date.fromisoformat(str(expected))
    except Exception:
        return None
    gap_cal = (d_exp - d_last).days
    gap_bus = _business_days_between(d_last, d_exp)
    return {
        "target_session": expected,
        "expected_session": expected,
        "effective_session": q.get("effective_session"),
        "last_date": last_date,
        "gap_days_calendar": gap_cal,
        "gap_days_business": gap_bus,
        "coverage_pct_effective": q.get("coverage_pct_effective"),
        "n_tickers_missing": q.get("n_tickers_missing_close_last"),
        "manifest_status": q.get("status"),
        "test_would_fail": bool(gap_bus > MAX_DAILY_AGE),
    }


def main() -> int:
    m = _load_manifest()
    row = _row_from_manifest(m)
    if row is None:
        print("[SKIP] manifest sin last_date o expected_session")
        return 0

    df_new = pd.DataFrame([row], columns=COLUMNS)
    if OUT_CSV.exists():
        df_old = pd.read_csv(OUT_CSV)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["target_session"], keep="last")
    else:
        df_all = df_new

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUT_CSV, index=False, lineterminator="\n")
    print("[OK] freshness_gap: target=" + str(row["target_session"])
          + " gap_bus=" + str(row["gap_days_business"])
          + " test_would_fail=" + str(row["test_would_fail"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
