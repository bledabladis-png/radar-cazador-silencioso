"""Regenera leader_representativeness.csv y sector_leader_divergence.csv.

2026-10-09 (reintento corregido): la primera version llamaba a
generate_leader_section con fase_dict={} lo que dejaba leader_df vacio
(porque el filtro VALID_FASES bloquea todo). El pipeline real construye
fase_dict desde compute_sector_scores(df_market). Este script replica
ese flujo sobre el parquet recortado a cada fecha.

Regenera las 13 fechas pre-08-oct. Conserva la 08-oct del snapshot pre
(publicada por el pipeline con v1.8).

NO productivo.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

import pandas as pd

from config.tickers import MARKET_TICKERS
from indicators.stock_leader import generate_leader_section
from indicators.leader_representativeness import compute_leader_representativeness
from indicators.sector_leader_divergence import compute_sector_leader_divergence
from regimes.sector_regime import compute_sector_scores

SECTORS = MARKET_TICKERS["sectors"]

STOCKS_PARQUET = Path("data/stock_prices.parquet")
MARKET_PARQUET = Path("data/market_data.parquet")
HOLDINGS_CSV = Path("data/etf_holdings.csv")
CONCENTRATION_CSV = Path("outputs/history/sector_concentration.csv")

REPR_CSV = Path("outputs/history/leader_representativeness.csv")
DIV_CSV = Path("outputs/history/sector_leader_divergence.csv")

REPR_PRE = Path("outputs/audit/leader_representativeness_pre_20261009.csv")
DIV_PRE = Path("outputs/audit/sector_leader_divergence_pre_20261009.csv")

META_COLS = ["effective_date", "expected_date", "coverage"]


def fases_from_market(df_market_cut):
    sr = compute_sector_scores(df_market_cut)
    if sr is None or "ranking" not in sr:
        return {}, {}
    fases = {sector: fase for sector, _, _, fase in sr["ranking"]}
    oper = {s: ("OPORTUNIDAD MODERADA" if f in ("ACCUMULATION", "MARKUP") else "NO OPERAR")
            for s, f in fases.items()}
    return fases, oper


def main():
    print("Cargando parquets...")
    df_stocks_full = pd.read_parquet(STOCKS_PARQUET)
    df_market_full = pd.read_parquet(MARKET_PARQUET)
    holdings_df = pd.read_csv(HOLDINGS_CSV)
    concentration_full = pd.read_csv(CONCENTRATION_CSV)
    concentration_full["date"] = pd.to_datetime(concentration_full["date"])

    print("Leyendo CSVs actuales...")
    repr_current = pd.read_csv(REPR_CSV)
    div_current = pd.read_csv(DIV_CSV)
    repr_current["date"] = pd.to_datetime(repr_current["date"])
    div_current["date"] = pd.to_datetime(div_current["date"])

    fechas = sorted(repr_current["date"].unique())
    print(f"Fechas: {len(fechas)}  ({fechas[0].date()} -> {fechas[-1].date()})")

    REPR_PRE.parent.mkdir(parents=True, exist_ok=True)
    repr_current.to_csv(REPR_PRE, index=False)
    div_current.to_csv(DIV_PRE, index=False)
    print("Snapshots guardados")

    last_date = fechas[-1]
    print(f"Ultima fecha (preserva): {last_date.date()}")

    repr_rows = []
    div_rows = []
    dates_with_leaders = 0

    for i, fecha in enumerate(fechas[:-1], 1):
        fecha_ts = pd.Timestamp(fecha)
        df_market_cut = df_market_full.loc[:fecha_ts]
        df_stocks_cut = df_stocks_full.loc[:fecha_ts]

        fases, oper = fases_from_market(df_market_cut)
        active = [s for s, f in fases.items() if f in ("ACCUMULATION", "MARKUP")]
        print(f"[{i}/{len(fechas)-1}] {fecha.date()} active_sectors={len(active)} {active}")

        if not active:
            continue

        result = generate_leader_section(
            df_market_cut, df_stocks_cut, holdings_df,
            fase_dict=fases, operabilidad_dict=oper,
            output_csv=None, temporal_meta=None,
        )
        if not isinstance(result, tuple) or len(result) != 3:
            print("  ERROR: retorno inesperado")
            sys.exit(1)
        _, leader_df, _ = result

        if leader_df is None or leader_df.empty:
            print(f"  leader_df vacio con {len(active)} sectores activos (inesperado)")
            continue

        dates_with_leaders += 1

        conc_cut = concentration_full[concentration_full["date"] <= fecha_ts].copy()
        if conc_cut.empty:
            print("  sin concentracion previa")
            continue
        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".csv")
        tmp_conc = Path(tmp_name)
        try:
            conc_cut.to_csv(tmp_conc, index=False)
            repr_df = compute_leader_representativeness(
                leader_df, str(tmp_conc), reference_date=fecha,
            )
            if repr_df is not None and not repr_df.empty:
                repr_rows.append(repr_df)
        finally:
            try:
                tmp_conc.unlink()
            except OSError:
                pass

        div_df = compute_sector_leader_divergence(
            df_stocks_cut, holdings_df, leader_df, df_market_cut,
            temporal_meta=None,
        )
        if div_df is not None and not div_df.empty:
            div_rows.append(div_df)

    if dates_with_leaders == 0:
        print("[ABORT] ninguna fecha produjo leader_df. No se escribe nada.")
        sys.exit(1)

    print()
    print(f"Fechas con leaders: {dates_with_leaders}")

    repr_regen = pd.concat(repr_rows, ignore_index=True) if repr_rows else pd.DataFrame()
    div_regen = pd.concat(div_rows, ignore_index=True) if div_rows else pd.DataFrame()

    repr_last = repr_current[repr_current["date"] == last_date].copy()
    div_last = div_current[div_current["date"] == last_date].copy()

    repr_final = pd.concat([repr_regen, repr_last], ignore_index=True)
    div_final = pd.concat([div_regen, div_last], ignore_index=True)

    if not div_regen.empty:
        meta_source = div_current[["date", "sector"] + META_COLS].copy()
        div_final = div_final.drop(columns=[c for c in META_COLS if c in div_final.columns], errors="ignore")
        div_final = div_final.merge(meta_source, on=["date", "sector"], how="left")

    repr_final = repr_final.sort_values(["date", "sector", "ticker"]).reset_index(drop=True)
    div_final = div_final.sort_values(["date", "sector"]).reset_index(drop=True)

    print(f"leader_representativeness final: {len(repr_final)} filas, {repr_final['date'].nunique()} fechas")
    print(f"sector_leader_divergence final: {len(div_final)} filas, {div_final['date'].nunique()} fechas")

    # Guardia: no escribir si la regeneracion dejo el CSV casi vacio.
    # El resultado esperado es ~115 repr / ~23 div (v1.8 filtra muchos
    # sectores que legacy marcaba como activos sin serlo). Umbral bajo
    # para detectar solo fallos catastroficos.
    if len(repr_final) < 60 or len(div_final) < 15:
        print(f"[ABORT] tamanos sospechosos: repr={len(repr_final)} div={len(div_final)}")
        sys.exit(1)

    tmp1 = REPR_CSV.with_suffix(REPR_CSV.suffix + ".tmp")
    tmp2 = DIV_CSV.with_suffix(DIV_CSV.suffix + ".tmp")
    repr_final.to_csv(tmp1, index=False, lineterminator="\n")
    div_final.to_csv(tmp2, index=False, lineterminator="\n")
    tmp1.replace(REPR_CSV)
    tmp2.replace(DIV_CSV)
    print(f"[OK] {REPR_CSV}")
    print(f"[OK] {DIV_CSV}")


if __name__ == "__main__":
    main()
