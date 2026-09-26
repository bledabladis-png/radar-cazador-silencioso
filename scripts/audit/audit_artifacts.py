"""Auditoria de integridad de artefactos - FASE 3.

Read-only. No modifica parquets ni manifests.
Uso: py scripts/audit_artifacts.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"

ARTIFACTS = [
    "market_data",
    "stock_prices",
    "commodities_futures",
    "commodities_spot",
    "cboe_vix3m",
]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(name: str) -> dict:
    p = DATA / f"{name}.parquet.manifest.json"
    return json.loads(p.read_text(encoding="utf-8-sig"))


def verify_hash(name: str):
    pq = DATA / f"{name}.parquet"
    mf = load_manifest(name)
    expected = mf["artifact"]["sha256"]
    actual = sha256_file(pq)
    ok = (expected == actual)
    print(f"[{name}] sha256 match: {ok}")
    if not ok:
        print(f"    expected: {expected}")
        print(f"    actual:   {actual}")
    return ok

def verify_content(name: str):
    pq = DATA / f"{name}.parquet"
    mf = load_manifest(name)
    c = mf.get("content", {})
    df = pd.read_parquet(pq)
    rows_ok = (len(df) == c.get("rows"))
    cols_ok = (df.shape[1] == c.get("cols"))
    print(f"[{name}] rows: declared={c.get('rows')} actual={len(df)} ok={rows_ok}")
    print(f"[{name}] cols: declared={c.get('cols')} actual={df.shape[1]} ok={cols_ok}")
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
        dmin = idx.min().date().isoformat()
        dmax = idx.max().date().isoformat()
        print(f"[{name}] index range: {dmin} -> {dmax}")
        print(f"    declared date_min={c.get('date_min')} date_max={c.get('date_max')}")
    return df

def check_futures_continuity():
    pq = DATA / "commodities_futures.parquet"
    df = pd.read_parquet(pq)
    print(f"[commodities_futures] shape={df.shape}")
    if isinstance(df.index, pd.DatetimeIndex):
        print(f"    fechas: {[d.date().isoformat() for d in df.index]}")
    elif "date" in df.columns:
        print(f"    fechas: {[d for d in df['date'].astype(str).unique()]}")
    else:
        print(f"    index: {type(df.index).__name__}, columns: {list(df.columns)}")
    return df


def check_spot_tail():
    pq = DATA / "commodities_spot.parquet"
    df = pd.read_parquet(pq)
    print(f"[commodities_spot] shape={df.shape}, columns={list(df.columns)}")
    if isinstance(df.index, pd.DatetimeIndex):
        tail = df.tail(6)
        print(tail.to_string())
    return df

def check_market_unknown():
    from src.instrument_registry import get_market
    pq = DATA / "market_data.parquet"
    df = pd.read_parquet(pq)
    print(f"[market_data] shape={df.shape}")

    if not isinstance(df.columns, pd.MultiIndex):
        print(f"    columns no es MultiIndex: {type(df.columns).__name__}")
        return

    tickers = sorted(set(df.columns.get_level_values(0)))
    unknown = [t for t in tickers if get_market(t) == "UNKNOWN"]
    print(f"    total tickers={len(tickers)}, UNKNOWN={len(unknown)}")
    for u in unknown:
        print(f"    UNKNOWN: {u}")

def main():
    print("=== VERIFICACION DE HASHES ===")
    for n in ARTIFACTS:
        verify_hash(n)

    print("\n=== VERIFICACION DE CONTENIDO ===")
    for n in ARTIFACTS:
        verify_content(n)

    print("\n=== FUTURES CONTINUIDAD ===")
    check_futures_continuity()

    print("\n=== SPOT TAIL ===")
    check_spot_tail()

    print("\n=== MARKET UNKNOWN ===")
    check_market_unknown()


if __name__ == "__main__":
    main()