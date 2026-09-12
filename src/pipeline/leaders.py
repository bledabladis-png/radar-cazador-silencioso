# -*- coding: utf-8 -*-
"""Fase 6a del pipeline: modulo de lideres sectoriales.

Carga df_stocks, valida cobertura, y genera leader_lines + leader_df +
full_metrics_df. Extraido de run.py (refactor C2, fase C2-7a).
"""

import pandas as pd

from src.stock_data_loader import download_stock_prices, get_usa_tickers
from src.utils import trim_to_last_valid_date, trim_to_last_valid_date_for_tickers


def compute_leaders(df_market, sector_results, reference_date=None):
    """Carga df_stocks y genera los lideres sectoriales.

    Returns:
        dict con keys:
            df_stocks (DataFrame | None), holdings_df (DataFrame | None),
            leader_lines (list | None), leader_df (DataFrame | None),
            full_metrics_df (DataFrame | None), holiday_mode (bool)
    """
    leader_lines = None
    df_stocks = None
    holdings_df = None
    leader_df = None
    full_metrics_df = None
    HOLIDAY_MODE = False

    try:
        df_stocks = download_stock_prices(reference_date=reference_date)
        if df_stocks is not None and not df_stocks.empty:
            try:
                usa_tickers = get_usa_tickers()
                if usa_tickers:
                    df_stocks = trim_to_last_valid_date_for_tickers(df_stocks, usa_tickers, min_coverage=0.8)
                else:
                    df_stocks = trim_to_last_valid_date(df_stocks)
            except Exception as e:
                print(f"  WARN usando trim generico: {e}")
                df_stocks = trim_to_last_valid_date(df_stocks)
            _close_cols = [c for c in df_stocks.columns if c[0] == 'Close']
            _n_close = len(_close_cols)
            if _n_close > 0:
                _last_valid = df_stocks[_close_cols].iloc[-1].notna().sum()
                _coverage_ratio = _last_valid / _n_close
                print(f"  DEBUG df_stocks Close columns: {_n_close}; shape={df_stocks.shape}; last_valid={_last_valid} ({_coverage_ratio:.2f})")
                if _coverage_ratio < 0.5:
                    print("  WARN Cobertura ultima fila insuficiente. Posible festivo. Se omitiran metricas dependientes de acciones.")
                    HOLIDAY_MODE = True
                else:
                    HOLIDAY_MODE = False
            else:
                HOLIDAY_MODE = True
            holdings_df = pd.read_csv('data/etf_holdings.csv')

            if HOLIDAY_MODE:
                # No usar df_stocks incompleto; los bloques dependientes se omiten
                df_stocks = None
            else:
                # Conservar df_stocks para calculo normal
                pass
            if HOLIDAY_MODE:
                leader_lines = None
                leader_df = None
                full_metrics_df = None
                print("  Lideres sectoriales omitidos (df_stocks no disponible).")
            else:
                fases = {sector: fase for sector, _, _, fase in sector_results['ranking']}
                oper = {sector: 'OPORTUNIDAD MODERADA' if fase in ['ACCUMULATION','MARKUP'] else 'NO OPERAR'
                        for sector, fase in fases.items()}
                from indicators.stock_leader import generate_leader_section
                leader_lines, leader_df, full_metrics_df = generate_leader_section(
                    df_market, df_stocks, holdings_df, fases, oper,
                    output_csv='outputs/report/analisis_lideres.csv'
                )
                if leader_lines:
                    print("  Lideres sectoriales generados.")
                else:
                    print("  No hay sectores favorables para lideres.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  Modulo de lideres omitido: {e}")

    return {
        'df_stocks': df_stocks,
        'holdings_df': holdings_df,
        'leader_lines': leader_lines,
        'leader_df': leader_df,
        'full_metrics_df': full_metrics_df,
    }
