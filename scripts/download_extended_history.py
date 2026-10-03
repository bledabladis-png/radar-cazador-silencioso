#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Descarga historico extendido 2015-09-26 para walk-forward SOW.

Contexto: el parquet productivo `data/stock_prices.parquet` cubre solo
2021-09-27 -> presente. El walk-forward anual de SOW necesita mas
regimenes (2015 China, 2018 Q4, 2020 COVID) para decidir si el
detector es robusto o dependiente de regimen.

Este script descarga el tramo anterior en un parquet SEPARADO
(`data/stock_prices_ext.parquet`). NO toca produccion.

Uso:
    py scripts/download_extended_history.py
    py scripts/download_extended_history.py --dry-run   # solo lista

Salida:
    data/stock_prices_ext.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.providers.yahoo import YahooProvider


# --- Rutas ---
PROD_PARQUET = ROOT / "data" / "stock_prices.parquet"
EXT_PARQUET = ROOT / "data" / "stock_prices_ext.parquet"

# --- Ventana de descarga (tramo anterior al parquet productivo) ---
START = "2015-01-02"
END = "2021-09-24"  # un viernes, anterior al 2021-09-27 del prod

# --- Batch (por resiliencia) ---
BATCH_SIZE = 50

def _tickers_from_prod():
    """Universo de tickers del parquet productivo (para mantener simetria)."""
    df = pd.read_parquet(PROD_PARQUET)
    close = df.xs("Close", axis=1, level=0)
    return sorted(close.columns.tolist())


def _download_batches(provider, tickers, start, end):
    """Descarga en lotes. Devuelve DataFrame concatenado o None."""
    frames = []
    total = len(tickers)
    for i in range(0, total, BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        print(f"  Batch {i // BATCH_SIZE + 1} "
              f"({i + 1}-{min(i + BATCH_SIZE, total)}/{total})...")
        t0 = time.time()
        try:
            df = provider.get_prices(batch, start=start, end=end)
            if df is not None and not df.empty:
                frames.append(df)
                print(f"    OK ({time.time()-t0:.1f}s, shape {df.shape})")
            else:
                print(f"    VACIO ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    ERROR: {e}")
    if not frames:
        return None
    return pd.concat(frames, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("== Descarga historico extendido ==")
    print(f"Ventana: {START} -> {END}")

    if not PROD_PARQUET.exists():
        print(f"[ABORT] No existe {PROD_PARQUET}")
        return 1

    tickers = _tickers_from_prod()
    print(f"Tickers a descargar: {len(tickers)}")

    if args.dry_run:
        print("Dry-run: no se descarga nada.")
        for tk in tickers[:10]:
            print(f"  {tk}")
        print(f"  ... y {len(tickers)-10} mas")
        return 0

    if EXT_PARQUET.exists():
        print(f"[WARN] {EXT_PARQUET} ya existe. No se sobreescribe.")
        print("       Borrar manualmente para regenerar.")
        return 0

    provider = YahooProvider()
    if not provider.is_available():
        print("[ABORT] Provider no disponible.")
        return 1

    print("== Descargando ==")
    t0 = time.time()
    df = _download_batches(provider, tickers, START, END)
    el = time.time() - t0

    if df is None or df.empty:
        print("[ABORT] Descarga vacia.")
        return 1

    print()
    print(f"Descarga completa en {el:.1f}s")
    print(f"Shape combinado: {df.shape}")
    print(f"Rango real: {df.index.min()} -> {df.index.max()}")
    print(f"Sesiones: {len(df.index)}")

    # Cobertura por ticker
    try:
        close = df.xs("Close", axis=1, level=0)
    except KeyError:
        close = df
    n_ok = (close.notna().sum() > 100).sum()
    print(f"Tickers con >100 sesiones validas: {n_ok}/{close.shape[1]}")

    print()
    print(f"Escribiendo {EXT_PARQUET}...")
    df.to_parquet(EXT_PARQUET)
    print(f"[OK] {EXT_PARQUET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())