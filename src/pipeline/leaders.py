# -*- coding: utf-8 -*-
"""Fase 6a del pipeline: modulo de lideres sectoriales.

Carga df_stocks, valida cobertura, y genera leader_lines + leader_df +
full_metrics_df. Extraido de run.py (refactor C2, fase C2-7a).
"""

import pandas as pd

from src.stock_data_loader import download_stock_prices
from src.effective_date import resolve_effective_date


def compute_leaders(df_market, sector_results, reference_date=None, run_id=None):
    """Carga df_stocks y genera los lideres sectoriales.

    Returns:
        dict con keys:
            df_stocks (DataFrame | None), holdings_df (DataFrame | None),
            leader_lines (list | None), leader_df (DataFrame | None),
            full_metrics_df (DataFrame | None), holiday_mode (bool)
    """
    leader_lines = None
    df_stocks = None
    df_stocks_effective_meta = None
    holdings_df = None
    leader_df = None
    full_metrics_df = None
    HOLIDAY_MODE = False

    try:
        df_stocks = download_stock_prices(reference_date=reference_date, run_id=run_id)
        if df_stocks is not None and not df_stocks.empty:
            # FU-020 (2026-09-15): sustitucion de trim_to_last_valid_date_for_tickers
            # por resolve_effective_date con universo completo (todas las columnas Close)
            # y min_coverage=0.90. El helper devuelve metadata (fecha efectiva, cobertura,
            # lag_dias) que se propaga al consumidor para que cada metrica agregada
            # declare su base temporal.
            _eligible = [c[1] for c in df_stocks.columns if c[0] == 'Close']
            _eff = resolve_effective_date(df_stocks, _eligible, min_coverage=0.90)
            if _eff["status"] == "OK" and _eff["date"] is not None:
                df_stocks = df_stocks.loc[:_eff["date"]]
                df_stocks_effective_meta = _eff
                print(f"  [FU-020] effective={pd.Timestamp(_eff['date']).date()} "
                      f"requested={pd.Timestamp(_eff['requested_date']).date()} "
                      f"lag={_eff['lag_days']}d "
                      f"coverage={_eff['coverage']:.2%} "
                      f"({_eff['n_observed']}/{_eff['n_eligible']})")
            else:
                print("  [FU-020] INSUFFICIENT_COVERAGE (min_coverage=0.90). "
                      "df_stocks omitido.")
                df_stocks = None
            if df_stocks is not None:
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
        'df_stocks_effective_meta': df_stocks_effective_meta,
        'holdings_df': holdings_df,
        'leader_lines': leader_lines,
        'leader_df': leader_df,
        'full_metrics_df': full_metrics_df,
    }
