#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Descarga cohorte reservada Russell 3000 (no en stock_prices_extended).

Produce: data/stock_prices_reserved_r3000.parquet (gitignored, ~100 MB).
Requiere: pip install yfinance pandas pyarrow requests

Uso: py scripts/fetch_reserved_cohort_r3000.py
"""
from __future__ import annotations
import io
import sys
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "stock_prices_reserved_r3000.parquet"
START = "2015-01-02"
END = "2026-10-24"
MIN_SESSIONS = 2000


def main() -> int:
    print("== Cohort reservada Russell 3000 ==")
    df_act = pd.read_parquet(ROOT / "data" / "stock_prices_extended.parquet")
    actuales = {str(t).upper() for t in df_act.xs("Close", axis=1, level=0).columns}
    print(f"Actuales: {len(actuales)}")

    # Intentar Russell 3000 desde GitHub
    tickers = set()
    urls = [
        "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt",
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                continue
            if url.endswith(".txt"):
                lines = r.text.strip().split("\n")
                tickers = {ln.strip().upper() for ln in lines if ln.strip() and "/" not in ln and "." not in ln[:1]}
            else:
                df = pd.read_csv(io.StringIO(r.text))
                col = "Symbol" if "Symbol" in df.columns else df.columns[0]
                tickers = {str(t).upper() for t in df[col]}
            print(f"OK desde {url.split('/')[-1]}: {len(tickers)}")
            break
        except Exception as e:
            print(f"  fallo {url}: {e}")
    if not tickers:
        print("ERROR: no se pudo descargar lista")
        return 1

    def norm(t: str) -> str:
        return t.replace("-", ".").replace(".", "-").upper()

    reservados = sorted({norm(t) for t in tickers} - {norm(t) for t in actuales})
    print(f"Reservados candidatos: {len(reservados)}")
    if len(reservados) < 100:
        print("ERROR: cohorte insuficiente")
        return 1

    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: pip install yfinance")
        return 1

    frames = {}
    batch = 50
    n_batches = (len(reservados) + batch - 1) // batch
    for i in range(0, len(reservados), batch):
        chunk = reservados[i:i + batch]
        print(f"  batch {i // batch + 1}/{n_batches}: {len(chunk)}")
        try:
            data = yf.download(chunk, start=START, end=END, group_by="ticker",
                               auto_adjust=False, progress=False, threads=True)
        except Exception as e:
            print(f"    fallo: {e}")
            continue
        if data.empty:
            continue
        for tk in chunk:
            try:
                sub = data[tk].dropna(how="all") if isinstance(data.columns, pd.MultiIndex) else data.dropna(how="all")
                if sub.empty or "Close" not in sub.columns:
                    continue
                if sub["Close"].notna().sum() < MIN_SESSIONS:
                    continue
                needed = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in sub.columns]
                frames[tk] = sub[needed]
            except (KeyError, TypeError):
                continue

    print(f"Validos: {len(frames)}")
    if len(frames) < 100:
        print("ERROR: cohorte resultante insuficiente")
        return 1

    combined = {}
    for field in ("Open", "High", "Low", "Close", "Volume"):
        sd = {tk: df[field] for tk, df in frames.items() if field in df.columns}
        if sd:
            combined[field] = pd.DataFrame(sd)
    out = pd.concat(combined, axis=1).sort_index()
    out.index.name = "Date"
    out.to_parquet(OUT)
    print(f"Guardado: {OUT} ({out.shape[0]} sesiones, {len(frames)} tickers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())