# -*- coding: utf-8 -*-
"""Fase 3 del pipeline: rankings sectoriales, precio/flujo, rotacion,
dispersion, correlacion y cross-asset.

Extraido de run.py (refactor C2, fase C2-5).
"""

from pathlib import Path

from src.utils import writer_observation_date
from src.pipeline.output_writer import persist, skip
from regimes.sector_regime import compute_sector_scores, compute_price_flow_rankings
from indicators.breadth import compute_breadth
from config.settings import EXPECTED_SECTOR_COUNT


def compute_sectors_base(df_market, temporal_meta=None):
    """Calcula los rankings y metricas sectoriales base.

    Returns:
        dict con keys:
            sector_results, sector_rank_deltas_df,
            sector_price_rank, sector_flow_rank, otros_price_rank, otros_flow_rank,
            sector_dispersion_df,
            sector_corr_summary_df,
            cross_asset_summary_df,
            breadth_values
    """
    print("Calculando rankings sectoriales...")
    # FU-021-5 Fase 5.1 (P2): fecha de observacion resuelta por contrato.
    _obs_date = writer_observation_date(
        temporal_meta, ['EQUITY_EOD'],
        lambda: df_market.index[-1])
    sector_results = compute_sector_scores(df_market)
    if sector_results:
        top3 = sector_results['ranking'][:3]
        print("  Top 3 sectores:")
        for i, (t, n, s) in enumerate(top3, 1):
            print(f"    {i}. {n} ({t}): {s:.2f}")
        print(f"  Regimen sectorial: {sector_results['regime']}")

    # --- Rotacion sectorial historica reciente v1.0 ---
    sector_rank_deltas_df = None
    try:
        from indicators.sector_rank_history import update_rank_history
        _, sector_rank_deltas_df = update_rank_history(
            sector_results, 'outputs/history/sector_rank_history.csv',
            date=_obs_date
        )
        if sector_rank_deltas_df is not None and not sector_rank_deltas_df.empty:
            print("  Rotacion sectorial historica calculada.")
            persist('sector_rank_deltas', sector_rank_deltas_df)
        else:
            skip('sector_rank_deltas', 'NO_DATA')
            sector_rank_deltas_df = None
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError) as e:
        print(f"  Rotacion sectorial omitida: {e}")
        skip('sector_rank_deltas', 'NO_DATA')
        sector_rank_deltas_df = None

    print("Calculando rankings de precio y flujo...")
    sector_price_rank, sector_flow_rank, otros_price_rank, otros_flow_rank = compute_price_flow_rankings(df_market)

    # --- Dispersion entre sectores v1.0 (descriptivo) ---
    sector_dispersion_df = None
    try:
        from indicators.sector_dispersion import compute_sector_dispersion
        sector_dispersion_df = compute_sector_dispersion(
            sector_price_rank, reference_date=_obs_date)
        sd_path = Path('outputs/history/sector_dispersion.csv')
        sd_path.parent.mkdir(parents=True, exist_ok=True)
        if not sector_dispersion_df.empty:
            date_val = sector_dispersion_df['date'].max()
            persist('sector_dispersion', sector_dispersion_df, date_value=date_val)
            print("  Dispersion entre sectores calculada.")
        else:
            skip('sector_dispersion', 'EMPTY_DF')
            sector_dispersion_df = None
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError) as e:
        print(f"  Dispersion entre sectores omitida: {e}")
        skip('sector_dispersion', 'NO_DATA')
        sector_dispersion_df = None

    # --- Correlacion entre sectores v1.1 (descriptivo) ---
    # F7-01: la matriz por pares fue eliminada (dead end sin consumidor).
    # Solo se persiste el summary (stats agregadas por ventana).
    sector_corr_summary_df = None
    try:
        from indicators.sector_correlation import compute_sector_correlation
        sector_corr_summary_df = compute_sector_correlation(df_market, temporal_meta=temporal_meta)
        cs_path = Path('outputs/history/sector_correlation_summary.csv')
        cs_path.parent.mkdir(parents=True, exist_ok=True)
        if not sector_corr_summary_df.empty:
            date_val = sector_corr_summary_df['date'].max()
            persist('sector_correlation_summary', sector_corr_summary_df, date_value=date_val)
            print("  Correlacion entre sectores calculada.")
        else:
            skip('sector_correlation_summary', 'EMPTY_DF')
            sector_corr_summary_df = None
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError) as e:
        print(f"  Correlacion entre sectores omitida: {e}")
        skip('sector_correlation_summary', 'NO_DATA')
        sector_corr_summary_df = None

    # --- Contexto Cross-Asset v1.1 (descriptivo) ---
    cross_asset_summary_df = None
    cross_asset_detail_df = None
    try:
        from indicators.cross_asset_context import compute_cross_asset_context
        cross_asset_detail_df, cross_asset_summary_df = compute_cross_asset_context(df_market, temporal_meta=temporal_meta)
        if not cross_asset_detail_df.empty:
            date_val = cross_asset_detail_df['date'].max()
            persist('cross_asset_correlation', cross_asset_detail_df, date_value=date_val)
        else:
            skip('cross_asset_correlation', 'EMPTY_DF')
        if not cross_asset_summary_df.empty:
            date_val = cross_asset_summary_df['date'].max()
            persist('cross_asset_context', cross_asset_summary_df, date_value=date_val)
            print("  Contexto Cross-Asset calculado.")
        else:
            skip('cross_asset_context', 'EMPTY_DF')
            cross_asset_summary_df = None
            cross_asset_detail_df = None
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError) as e:
        print(f"  Contexto Cross-Asset omitido: {e}")
        skip('cross_asset_correlation', 'NO_DATA')
        skip('cross_asset_context', 'NO_DATA')
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
        'EMA20 count': int(round(b20.iloc[-1] * EXPECTED_SECTOR_COUNT)),
        'EMA50 count': int(round(b50.iloc[-1] * EXPECTED_SECTOR_COUNT)),
        'EMA200 count': int(round(b200.iloc[-1] * EXPECTED_SECTOR_COUNT)),
        'New Highs count': int(round(nh.iloc[-1] * EXPECTED_SECTOR_COUNT)),
        'New Lows count': int(round(nl.iloc[-1] * EXPECTED_SECTOR_COUNT)),
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
        'cross_asset_summary_df': cross_asset_summary_df,
        'breadth_values': breadth_values,
    }
