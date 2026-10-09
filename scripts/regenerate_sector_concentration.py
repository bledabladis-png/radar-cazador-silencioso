"""Regenera sector_concentration.csv con v1.8 retroactivo.

2026-10-09: el CSV acumula 187 filas pre-08-oct calculadas con el
clasificador legacy (robust_zscore(trend, w=60)) y 11 filas post-08-oct
con v1.8 (tanh(trend/K)). Dos metricas bajo el mismo nombre:
- wyckoff_median
- wls_median (derivada, usa wyckoff_score en el calculo de WLS)

Este script reconstruye las 18 fechas del CSV aplicando v1.8 sobre el
parquet recortado a cada fecha. Mismo patron que
regenerate_wyckoff_distribution.py (commit f782b9d4).

NO productivo. Se ejecuta una vez.
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd

from config.tickers import MARKET_TICKERS
from indicators.sector_concentration import compute_sector_concentration
from indicators.stock_leader import generate_leader_section

SECTORS = MARKET_TICKERS["sectors"]

CSV_PATH = Path("outputs/history/sector_concentration.csv")
SNAPSHOT_PRE = Path("outputs/audit/sector_concentration_pre_20261009.csv")
OUT_TMP = CSV_PATH.with_suffix(CSV_PATH.suffix + ".tmp")

STOCKS_PARQUET = Path("data/stock_prices.parquet")
MARKET_PARQUET = Path("data/market_data.parquet")
HOLDINGS_CSV = Path("data/etf_holdings.csv")


def main():
    print(f"Cargando {STOCKS_PARQUET}...")
    df_stocks_full = pd.read_parquet(STOCKS_PARQUET)
    print(f"  shape: {df_stocks_full.shape}")
    print(f"Cargando {MARKET_PARQUET}...")
    df_market_full = pd.read_parquet(MARKET_PARQUET)
    print(f"  shape: {df_market_full.shape}")
    print(f"Cargando {HOLDINGS_CSV}...")
    holdings_df = pd.read_csv(HOLDINGS_CSV)

    print(f"Leyendo fechas de {CSV_PATH}...")
    current = pd.read_csv(CSV_PATH)
    fechas = sorted(current["date"].unique())
    print(f"  fechas a regenerar: {len(fechas)}")

    SNAPSHOT_PRE.parent.mkdir(parents=True, exist_ok=True)
    current.to_csv(SNAPSHOT_PRE, index=False)
    print(f"Snapshot pre: {SNAPSHOT_PRE}")
    print()

    rows = []
    for i, fecha in enumerate(fechas, 1):
        print(f"[{i}/{len(fechas)}] {fecha}...")
        fecha_ts = pd.Timestamp(fecha)
        df_market_cut = df_market_full.loc[:fecha_ts]
        df_stocks_cut = df_stocks_full.loc[:fecha_ts]

        result = generate_leader_section(
            df_market_cut, df_stocks_cut, holdings_df,
            fase_dict={}, operabilidad_dict={},
            output_csv=None, temporal_meta=None,
        )
        # generate_leader_section devuelve (lines, leader_df, full_metrics_df)
        if not isinstance(result, tuple) or len(result) != 3:
            print(f"  ERROR: retorno inesperado {type(result)}")
            sys.exit(1)
        _, _, full_metrics_df = result

        if full_metrics_df is None or full_metrics_df.empty:
            print(f"  sin full_metrics para {fecha}")
            continue

        sc_df = compute_sector_concentration(
            df_stocks_cut, holdings_df, full_metrics_df,
            reference_date=fecha,
        )
        if sc_df is None or sc_df.empty:
            print(f"  sin concentracion para {fecha}")
            continue
        rows.append(sc_df)

    if not rows:
        print("ERROR: sin filas generadas")
        sys.exit(1)

    result = pd.concat(rows, ignore_index=True)
    result = result[result["sector"].isin(SECTORS)].copy()
    print()
    print(f"Total filas generadas: {len(result)}")
    print(f"Fechas en resultado: {result['date'].nunique()}")
    print(f"Sectores en resultado: {result['sector'].nunique()}")

    result.to_csv(OUT_TMP, index=False, lineterminator="\n")
    OUT_TMP.replace(CSV_PATH)
    print(f"Escrito {CSV_PATH}")


if __name__ == "__main__":
    main()
