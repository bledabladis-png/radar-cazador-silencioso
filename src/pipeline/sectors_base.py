# -*- coding: utf-8 -*-
"""Fase 3 del pipeline: rankings sectoriales, precio/flujo, rotacion,
dispersion, correlacion y cross-asset.

Extraido de run.py (refactor C2, fase C2-5).
"""

from pathlib import Path

import pandas as pd

from src.utils import append_dedup
from regimes.sector_regime import compute_sector_scores, compute_price_flow_rankings
from indicators.breadth import compute_breadth


def compute_sectors_base(df_market):
    """Calcula los rankings y metricas sectoriales base.

    Returns:
        dict con keys:
            sector_results, sector_rank_deltas_df,
            sector_price_rank, sector_flow_rank, otros_price_rank, otros_flow_rank,
            sector_dispersion_df,
            sector_corr_summary_df, sector_corr_matrix_df,
            cross_asset_summary_df, cross_asset_detail_df,
            breadth_values
    """
    print("Calculando rankings sectoriales...")
    sector_results = compute_sector_scores(df_market)
    if sector_results:
        top3 = sector_results['ranking'][:3]
        print("  Top 3 sectores:")
        for i, (t, n, s, w) in enumerate(top3, 1):
            print(f"    {i}. {n} ({t}): {s:.2f} [{w}]")
        print(f"  Regimen sectorial: {sector_results['regime']}")

    # --- Rotacion sectorial historica reciente v1.0 ---
    sector_rank_deltas_df = None
    try:
        from indicators.sector_rank_history import update_rank_history
        _, sector_rank_deltas_df = update_rank_history(
            sector_results, 'outputs/history/sector_rank_history.csv', date=pd.Timestamp.now().normalize()
        )
        if sector_rank_deltas_df is not None and not sector_rank_deltas_df.empty:
            print("  Rotacion sectorial historica calculada.")
            srd_path = Path('outputs/history/sector_rank_deltas.csv')
            srd_path.parent.mkdir(parents=True, exist_ok=True)
            sector_rank_deltas_df.to_csv(srd_path, index=False, encoding='utf-8')
        else:
            sector_rank_deltas_df = None
    except Exception as e:
        print(f"  Rotacion sectorial omitida: {e}")
        sector_rank_deltas_df = None

    print("Calculando rankings de precio y flujo...")
    sector_price_rank, sector_flow_rank, otros_price_rank, otros_flow_rank = compute_price_flow_rankings(df_market)

    # --- Dispersion entre sectores v1.0 (descriptivo) ---
    sector_dispersion_df = None
    try:
        from indicators.sector_dispersion import compute_sector_dispersion
        sector_dispersion_df = compute_sector_dispersion(sector_price_rank)
        sd_path = Path('outputs/history/sector_dispersion.csv')
        sd_path.parent.mkdir(parents=True, exist_ok=True)
        if not sector_dispersion_df.empty:
            if sd_path.exists():
                hist_sd = pd.read_csv(sd_path)
                sector_dispersion_df = append_dedup(hist_sd, sector_dispersion_df, ['date'])
            sector_dispersion_df.to_csv(sd_path, index=False)
            print("  Dispersion entre sectores calculada.")
        else:
            sector_dispersion_df = None
    except Exception as e:
        print(f"  Dispersion entre sectores omitida: {e}")
        sector_dispersion_df = None

    # --- Correlacion entre sectores v1.0 (descriptivo) ---
    sector_corr_summary_df = None
    sector_corr_matrix_df = None
    try:
        from indicators.sector_correlation import compute_sector_correlation
        sector_corr_matrix_df, sector_corr_summary_df = compute_sector_correlation(df_market)
        cm_path = Path('outputs/history/sector_correlation_matrix.csv')
        cs_path = Path('outputs/history/sector_correlation_summary.csv')
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        if not sector_corr_matrix_df.empty:
            if cm_path.exists():
                hist_cm = pd.read_csv(cm_path)
                sector_corr_matrix_df = append_dedup(hist_cm, sector_corr_matrix_df, ['date','window','sector1','sector2'])
            sector_corr_matrix_df.to_csv(cm_path, index=False)
        if not sector_corr_summary_df.empty:
            if cs_path.exists():
                hist_cs = pd.read_csv(cs_path)
                sector_corr_summary_df = append_dedup(hist_cs, sector_corr_summary_df, ['date','window'])
            sector_corr_summary_df.to_csv(cs_path, index=False)
            print("  Correlacion entre sectores calculada.")
        else:
            sector_corr_summary_df = None
            sector_corr_matrix_df = None
    except Exception as e:
        print(f"  Correlacion entre sectores omitida: {e}")
        sector_corr_summary_df = None

    # --- Contexto Cross-Asset v1.1 (descriptivo) ---
    cross_asset_summary_df = None
    cross_asset_detail_df = None
    try:
        from indicators.cross_asset_context import compute_cross_asset_context
        cross_asset_detail_df, cross_asset_summary_df = compute_cross_asset_context(df_market)
        ca_detail_path = Path('outputs/history/cross_asset_correlation.csv')
        ca_summary_path = Path('outputs/history/cross_asset_context.csv')
        ca_detail_path.parent.mkdir(parents=True, exist_ok=True)
        if not cross_asset_detail_df.empty:
            if ca_detail_path.exists():
                hist_cad = pd.read_csv(ca_detail_path)
                cross_asset_detail_df = append_dedup(hist_cad, cross_asset_detail_df, ['date','window','sector','asset'])
            cross_asset_detail_df.to_csv(ca_detail_path, index=False)
        if not cross_asset_summary_df.empty:
            if ca_summary_path.exists():
                hist_cas = pd.read_csv(ca_summary_path)
                cross_asset_summary_df = append_dedup(hist_cas, cross_asset_summary_df, ['date','window','sector','asset_class'])
            cross_asset_summary_df.to_csv(ca_summary_path, index=False)
            print("  Contexto Cross-Asset calculado.")
        else:
            cross_asset_summary_df = None
            cross_asset_detail_df = None
    except Exception as e:
        print(f"  Contexto Cross-Asset omitido: {e}")
        cross_asset_summary_df = None
        cross_asset_detail_df = None

    # Breadth ampliado
    b20, b50, b200, nh, nl = compute_breadth(df_market)

    breadth_values = {
        '% sobre EMA20': b20.iloc[-1],
        '% sobre EMA50': b50.iloc[-1],
        '% sobre EMA200': b200.iloc[-1],
        'New Highs (%)': nh.iloc[-1],
        'New Lows (%)': nl.iloc[-1],
        'EMA20 count': int(round(b20.iloc[-1] * 11)),
        'EMA50 count': int(round(b50.iloc[-1] * 11)),
        'EMA200 count': int(round(b200.iloc[-1] * 11)),
        'New Highs count': int(round(nh.iloc[-1] * 11)),
        'New Lows count': int(round(nl.iloc[-1] * 11)),
    }

    return {
        'sector_results': sector_results,
        'sector_rank_deltas_df': sector_rank_deltas_df,
        'sector_price_rank': sector_price_rank,
        'sector_flow_rank': sector_flow_rank,
        'otros_price_rank': otros_price_rank,
        'otros_flow_rank': otros_flow_rank,
        'sector_dispersion_df': sector_dispersion_df,
        'sector_corr_summary_df': sector_corr_summary_df,
        'sector_corr_matrix_df': sector_corr_matrix_df,
        'cross_asset_summary_df': cross_asset_summary_df,
        'breadth_values': breadth_values,
    }
