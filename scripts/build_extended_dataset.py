#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fusiona stock_prices.parquet + stock_prices_ext.parquet para el
walk-forward extendido de SOW. NO toca produccion.

Uso:
    py scripts/build_extended_dataset.py

Salida:
    data/stock_prices_extended.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROD = ROOT / "data" / "stock_prices.parquet"
EXT = ROOT / "data" / "stock_prices_ext.parquet"
OUT = ROOT / "data" / "stock_prices_extended.parquet"


def main():
    if not PROD.exists():
        print(f"[ABORT] No existe {PROD}")
        return 1
    if not EXT.exists():
        print(f"[ABORT] No existe {EXT}")
        return 1

    print(f"== Cargando productivo: {PROD.name} ==")
    df_prod = pd.read_parquet(PROD)
    print(f"  shape: {df_prod.shape}")
    print(f"  rango: {df_prod.index.min()} -> {df_prod.index.max()}")

    print(f"== Cargando extendido: {EXT.name} ==")
    df_ext = pd.read_parquet(EXT)
    print(f"  shape: {df_ext.shape}")
    print(f"  rango: {df_ext.index.min()} -> {df_ext.index.max()}")

    print("== Fusionando ==")
    # Alinear columnas (union de ambas)
    cols_union = df_prod.columns.union(df_ext.columns)
    df_prod = df_prod.reindex(columns=cols_union)
    df_ext = df_ext.reindex(columns=cols_union)
    df = pd.concat([df_ext, df_prod], axis=0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    print(f"  shape: {df.shape}")
    print(f"  rango: {df.index.min()} -> {df.index.max()}")
    print(f"  sesiones: {len(df.index)}")

    # Cobertura por año
    close = df.xs("Close", axis=1, level=0)
    by_year = close.groupby(close.index.year).apply(
        lambda g: g.notna().sum().mean() / len(g)
    )
    print()
    print("Cobertura media por año (Close no-NaN):")
    for year, cov in by_year.items():
        print(f"  {year}: {cov:.3f}")

    print()
    print(f"Escribiendo {OUT}...")
    df.to_parquet(OUT)
    print(f"[OK] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())