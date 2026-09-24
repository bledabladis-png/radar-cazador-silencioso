"""Genera mapa CUSIP -> FIGI del radar desde filings + catalogo.

Salida: data/mappings/cusip_to_radar_figi.csv
Columnas: cusip, radar_ticker, share_class_figi, n_filings, emisor

Determinista. Sin red. Sin datetime.now().
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

OUT = ROOT / "data" / "mappings" / "cusip_to_radar_figi.csv"


def build_map(quarter: str) -> pd.DataFrame:
    """Un CUSIP -> (radar_ticker, share_class_figi) usando FIGIs del radar."""
    cat = pd.read_csv(ROOT / "data/mappings/radar_target_catalog.csv", dtype=str)
    cat_ok = cat[cat["status"] == "OK"][["radar_ticker", "share_class_figi"]].dropna()
    scf_to_ticker = dict(zip(cat_ok["share_class_figi"], cat_ok["radar_ticker"]))
    radar_scf = set(scf_to_ticker.keys())

    df = pd.read_parquet(
        ROOT / f"data/sec_13f/processed/{quarter}/INFOTABLE.parquet",
        columns=["CUSIP", "FIGI", "NAMEOFISSUER"],
    )
    df = df[df["FIGI"].notna()].copy()
    df = df[df["FIGI"].isin(radar_scf)].copy()

    # Agrupamos: un FIGI por CUSIP (el mas frecuente)
    mapa = (
        df.groupby(["CUSIP", "FIGI"])
        .agg(n_filings=("FIGI", "count"), emisor=("NAMEOFISSUER", "first"))
        .reset_index()
        .sort_values(["CUSIP", "n_filings"], ascending=[True, False])
        .drop_duplicates(subset=["CUSIP"], keep="first")
    )
    mapa["radar_ticker"] = mapa["FIGI"].map(scf_to_ticker)
    mapa = mapa.rename(columns={"FIGI": "share_class_figi", "CUSIP": "cusip"})
    mapa = mapa[["cusip", "radar_ticker", "share_class_figi", "n_filings", "emisor"]]
    return mapa.sort_values("radar_ticker").reset_index(drop=True)


def main():
    q4 = build_map("2025Q4")
    q1 = build_map("2026Q1")

    # Union: un CUSIP -> FIGI debe ser estable entre trimestres
    union = pd.concat([q4, q1], ignore_index=True)
    union = union.sort_values(["cusip", "n_filings"], ascending=[True, False])
    union = union.drop_duplicates(subset=["cusip"], keep="first")
    union = union.sort_values("radar_ticker").reset_index(drop=True)

    OUT.write_text(union.to_csv(index=False, lineterminator="\n"), encoding="utf-8")

    print(f"OK -> {OUT}")
    print(f"  Filas: {len(union)}")
    print(f"  Tickers unicos: {union['radar_ticker'].nunique()}")
    print()
    print("Top 15:")
    print(union.head(15).to_string())


if __name__ == "__main__":
    main()