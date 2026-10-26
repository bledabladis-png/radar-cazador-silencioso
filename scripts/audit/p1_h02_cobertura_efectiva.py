"""P1-H02: cobertura efectiva por grupo (dictamen 2026-10-26).

Cuantifica la diferencia entre:
  - seleccionado: los N tickers que devuelve _select_target_holdings.
  - con datos: el ticker esta en stock_prices.parquet con n_bars >= 200
    (MIN_BARS_THEORETICAL). Es el filtro primario de _evaluate_holding.
  - evaluable: fila en compute_wyckoff_snapshot. Requiere classify_phase.
    Este script NO lo calcula (classify_phase cuelga con al menos un
    ticker, bug conocido). Se reporta con_datos como proxy y se declara
    la limitacion en columna 'evaluable_metodo'.
  - publicado: fila en alguno de los 6 CSVs de outputs/history/.

Cobertura ponderada (peso del ticker, no conteo):
  - cov_vs_sel_pct: con_datos_peso / seleccionado_peso.
  - cov_vs_grupo_pct: con_datos_peso / weight_total del grupo.

Read-only. No modifica nada. Escribe CSVs en outputs/audit/ (gitignored).
Uso: py scripts/audit/p1_h02_cobertura_efectiva.py
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from indicators.wyckoff_holdings_report import (
    _select_target_holdings, get_col, MIN_BARS_THEORETICAL,
)
from src.instrument_registry import normalize_yahoo_ticker
from src.pipeline.wyckoff_holdings import (
    _load_index_holdings, _split_indices,
)

DATA = ROOT / "data"
HIST = ROOT / "outputs" / "history"
OUT = ROOT / "outputs" / "audit" / "p1_h02_cobertura_efectiva.csv"

CSVS = [
    ("wyckoff_usa_sector_accumulation", "sector_usa"),
    ("wyckoff_usa_sector_distribution", "sector_usa"),
    ("wyckoff_usa_index_accumulation", "index_usa"),
    ("wyckoff_usa_index_distribution", "index_usa"),
    ("wyckoff_eu_index_accumulation", "index_eu"),
    ("wyckoff_eu_index_distribution", "index_eu"),
]


def _has_data(df_stocks, ticker):
    t = normalize_yahoo_ticker(ticker)
    try:
        s = get_col(df_stocks, t, "Close").dropna()
        return len(s) >= MIN_BARS_THEORETICAL, int(len(s)), t
    except (KeyError, ValueError, TypeError):
        return False, 0, t


def _published_by_group():
    pub = {}
    for name, gt in CSVS:
        p = HIST / (name + ".csv")
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "group" not in df.columns or "weight" not in df.columns:
            continue
        for g, sub in df.groupby("group"):
            key = (gt, g)
            rec = pub.setdefault(key, {"n": 0, "peso": 0.0})
            rec["n"] += int(len(sub))
            rec["peso"] += float(sub["weight"].sum())
    return pub


def main():
    df_stocks = pd.read_parquet(DATA / "stock_prices.parquet")
    etf_h = pd.read_csv(DATA / "etf_holdings.csv")
    usa_idx, eu_idx = _split_indices(_load_index_holdings())
    pub = _published_by_group()

    rows = []
    sin_datos = []
    for df_h, gt in ((etf_h, "sector_usa"),
                     (usa_idx, "index_usa"),
                     (eu_idx, "index_eu")):
        if df_h is None or df_h.empty:
            continue
        for g, sub in df_h.groupby("etf"):
            wt = float(sub["weight"].sum())
            sel, _, estado = _select_target_holdings(sub)
            sel_peso = float(sel["weight"].sum()) if len(sel) else 0.0
            cd_n, cd_peso = 0, 0.0
            for _, r in sel.iterrows():
                ok, nb, tn = _has_data(df_stocks, r["ticker"])
                if ok:
                    cd_n += 1
                    cd_peso += float(r["weight"])
                else:
                    sin_datos.append({
                        "group": g, "gt": gt,
                        "ticker": r["ticker"], "ticker_norm": tn,
                        "weight": float(r["weight"]), "n_bars": nb,
                    })
            pr = pub.get((gt, g), {"n": 0, "peso": 0.0})
            rows.append({
                "group": g, "gt": gt, "estado": estado,
                "wt": round(wt, 2),
                "sel_n": len(sel), "sel_peso": round(sel_peso, 2),
                "cd_n": cd_n, "cd_peso": round(cd_peso, 2),
                "ev_n_proxy": cd_n, "ev_peso_proxy": round(cd_peso, 2),
                "evaluable_metodo": "proxy=con_datos (classify_phase omitido)",
                "pub_n": pr["n"], "pub_peso": round(pr["peso"], 2),
                "cov_vs_sel_pct": round(cd_peso / sel_peso * 100, 2) if sel_peso else 0.0,
                "cov_vs_grupo_pct": round(cd_peso / wt * 100, 2) if wt else 0.0,
            })

    df = pd.DataFrame(rows).sort_values(["gt", "group"])
    print("=" * 100)
    print("P1-H02: COBERTURA EFECTIVA POR GRUPO")
    print("=" * 100)
    print(df.to_string(index=False))
    print()
    print("Total grupos:", len(df))
    print("Nota: 'ev_*' es proxy (con_datos); classify_phase omitido por hang conocido.")
    print()

    df_sd = pd.DataFrame(sin_datos)
    if len(df_sd):
        print("=" * 100)
        print("TICKERS SIN DATOS EN PRODUCCION (universo real normalizado)")
        print("=" * 100)
        print(df_sd.sort_values(["group", "weight"], ascending=[True, False]).to_string(index=False))
        print()
        print("Total:", len(df_sd))
        print()
        print("Impacto en cobertura por grupo:")
        for g, sub in df_sd.groupby("group"):
            wt = float(next(r["wt"] for r in rows if r["group"] == g))
            sel_peso = float(next(r["sel_peso"] for r in rows if r["group"] == g))
            print("  {:8s} peso_faltante={:5.2f}  sobre_grupo={:5.1f}%  sobre_seleccion={:5.1f}%".format(
                g, sub["weight"].sum(),
                sub["weight"].sum() / wt * 100,
                sub["weight"].sum() / sel_peso * 100 if sel_peso else 0.0))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    if len(df_sd):
        sd_out = OUT.with_name("p1_h02_tickers_sin_datos.csv")
        df_sd.to_csv(sd_out, index=False)
        print()
        print("escrito:", OUT)
        print("escrito:", sd_out)
    else:
        print()
        print("escrito:", OUT)


if __name__ == "__main__":
    main()
