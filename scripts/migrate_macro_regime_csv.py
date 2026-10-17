# -*- coding: utf-8 -*-
"""Migracion del CSV de regimenes al schema post-commit-B.

REP-AUDIT-20261017-capa6-commitD (2026-10-17).

Antes de este commit, la columna "liquidity_regime" contenia el
regimen FINANCIERO (financial_conditions.py), no el de liquidez real
FRED (liquidity_v2.py). Bug de naming documentado en commit B.

Este script migra el CSV:
- Si solo hay "liquidity_regime": renombra a "financial_regime".
- Si hay ambas: verifica igualdad, elimina "liquidity_regime".
- Si no hay "real_liq_regime": anade con NaN (sin imputacion).

Idempotente: ejecutarlo dos veces no cambia el resultado.

Uso:
    py scripts/migrate_macro_regime_csv.py [--dry-run]
"""
import sys
from pathlib import Path

import pandas as pd

PATH = Path("outputs/history/macro_regime.csv")


def migrate(path, dry_run=False):
    """Migra el CSV. Devuelve dict con status y cambios aplicados."""
    p = Path(path)
    if not p.exists():
        return {"status": "not_found", "path": str(p)}

    df = pd.read_csv(p)
    cambios = []

    if "liquidity_regime" in df.columns and "financial_regime" not in df.columns:
        df = df.rename(columns={"liquidity_regime": "financial_regime"})
        cambios.append("renamed liquidity_regime -> financial_regime")

    elif "liquidity_regime" in df.columns and "financial_regime" in df.columns:
        mask = df["liquidity_regime"].notna() & df["financial_regime"].notna()
        if mask.any() and not (df.loc[mask, "liquidity_regime"] == df.loc[mask, "financial_regime"]).all():
            raise ValueError(
                "liquidity_regime y financial_regime difieren en al menos una fila. "
                "Migracion ambigua, abortar."
            )
        df = df.drop(columns=["liquidity_regime"])
        cambios.append("dropped liquidity_regime (identical to financial_regime)")

    if "real_liq_regime" not in df.columns:
        df["real_liq_regime"] = pd.NA
        cambios.append("added real_liq_regime=NaN (historic rows, no imputation)")

    if not cambios:
        return {"status": "already_migrated", "path": str(p)}

    if not dry_run:
        df.to_csv(p, index=False)

    return {
        "status": "migrated" if not dry_run else "dry_run",
        "cambios": cambios,
        "rows": len(df),
        "columns": list(df.columns),
        "path": str(p),
    }


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    result = migrate(PATH, dry_run=dry)
    for k, v in result.items():
        print("{0}: {1}".format(k, v))
