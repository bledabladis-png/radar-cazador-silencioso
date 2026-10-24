# -*- coding: utf-8 -*-
"""Fase 6a del pipeline: modulo de lideres sectoriales.

C4.2: la carga de df_stocks/holdings se extrajo a stocks_data.py.
Este modulo solo genera leader_lines + leader_df + full_metrics_df
a partir de esos datos. Se retira completo en C4.3.
"""
from src.pipeline.stocks_data import load_stocks_data


def compute_leaders(df_market, sector_results, reference_date=None, run_id=None, temporal_meta=None):
    """Carga df_stocks y genera los lideres sectoriales.

    Returns:
        dict con keys:
            df_stocks (DataFrame | None),
            df_stocks_effective_meta (dict | None),
            holdings_df (DataFrame | None),
            leader_lines (list | None), leader_df (DataFrame | None),
            full_metrics_df (DataFrame | None)
    """
    leader_lines = None
    leader_df = None
    full_metrics_df = None

    sd = load_stocks_data(reference_date=reference_date, run_id=run_id)
    df_stocks = sd["df_stocks"]
    df_stocks_effective_meta = sd["df_stocks_effective_meta"]
    holdings_df = sd["holdings_df"]

    if holdings_df is None:
        return {
            "df_stocks": df_stocks,
            "df_stocks_effective_meta": df_stocks_effective_meta,
            "holdings_df": None,
            "leader_lines": None,
            "leader_df": None,
            "full_metrics_df": None,
        }

    if df_stocks is None:
        return {
            "df_stocks": None,
            "df_stocks_effective_meta": df_stocks_effective_meta,
            "holdings_df": holdings_df,
            "leader_lines": None,
            "leader_df": None,
            "full_metrics_df": None,
        }

    try:
        fases = {sector: fase for sector, _, _, fase in sector_results["ranking"]}
        oper = {sector: "OPORTUNIDAD MODERADA" if fase in ["ACCUMULATION", "MARKUP"] else "NO OPERAR"
                for sector, fase in fases.items()}
        from indicators.stock_leader import generate_leader_section
        leader_lines, leader_df, full_metrics_df = generate_leader_section(
            df_market, df_stocks, holdings_df, fases, oper,
            output_csv="outputs/report/analisis_lideres.csv",
            temporal_meta=temporal_meta,
        )
        if leader_lines:
            print("  Lideres sectoriales generados.")
        else:
            print("  No hay sectores favorables para lideres.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  Seleccion de lideres omitida: {e}")

    return {
        "df_stocks": df_stocks,
        "df_stocks_effective_meta": df_stocks_effective_meta,
        "holdings_df": holdings_df,
        "leader_lines": leader_lines,
        "leader_df": leader_df,
        "full_metrics_df": full_metrics_df,
    }
