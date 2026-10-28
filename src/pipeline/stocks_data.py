# -*- coding: utf-8 -*-
"""Carga de df_stocks + holdings para el pipeline.

Extraido de leaders.py::compute_leaders (C4.2). Responsabilidades:
  - Descargar / cachear df_stocks.
  - Resolver effective_date (FU-020, min_coverage=0.90).
  - Aplicar HOLIDAY_MODE (cobertura ultima fila < 50%).
  - Cargar data/etf_holdings.csv (holdings_df).

No genera lideres. La seleccion de lideres se retira en C4.3.
"""
import pandas as pd

from src.stock_data_loader import download_stock_prices
from src.effective_date import resolve_effective_date

ETF_HOLDINGS_PATH = "data/etf_holdings.csv"
HOLIDAY_MODE_COVERAGE_THRESHOLD = 0.5


def load_stocks_data(reference_date=None, run_id=None):
    """Carga df_stocks, effective_meta y holdings_df.

    Returns:
        dict con keys:
            df_stocks (DataFrame | None),
            df_stocks_effective_meta (dict | None),
            holdings_df (DataFrame | None)

    df_stocks es None si no hay datos o si la cobertura de la ultima
    fila cae por debajo del 50 por ciento (modo festivo).
    """
    df_stocks = None
    df_stocks_effective_meta = None
    holdings_df = None
    HOLIDAY_MODE = False

    try:
        df_stocks = download_stock_prices(reference_date=reference_date, run_id=run_id)
        if df_stocks is not None and not df_stocks.empty:
            _eligible = [c[1] for c in df_stocks.columns if c[0] == "Close"]
            _eff = resolve_effective_date(df_stocks, _eligible)
            if _eff["status"] == "OK" and _eff["date"] is not None:
                df_stocks = df_stocks.loc[:_eff["date"]]
                df_stocks_effective_meta = _eff
                print(
                    f"  [FU-020] effective={pd.Timestamp(_eff['date']).date()} "
                    f"requested={pd.Timestamp(_eff['requested_date']).date()} "
                    f"lag={_eff['lag_days']}d "
                    f"coverage={_eff['coverage']:.2%} "
                    f"({_eff['n_observed']}/{_eff['n_eligible']})"
                )
            else:
                print("  [FU-020] INSUFFICIENT_COVERAGE (min_coverage=0.90). df_stocks omitido.")
                df_stocks = None
            if df_stocks is not None:
                _close_cols = [c for c in df_stocks.columns if c[0] == "Close"]
                _n_close = len(_close_cols)
                if _n_close > 0:
                    _last_valid = df_stocks[_close_cols].iloc[-1].notna().sum()
                    _coverage_ratio = _last_valid / _n_close
                    print(f"  DEBUG df_stocks Close columns: {_n_close}; shape={df_stocks.shape}; last_valid={_last_valid} ({_coverage_ratio:.2f})")
                    if _coverage_ratio < HOLIDAY_MODE_COVERAGE_THRESHOLD:
                        print("  WARN Cobertura ultima fila insuficiente. Posible festivo. Se omitiran metricas dependientes de acciones.")
                        HOLIDAY_MODE = True
                    else:
                        HOLIDAY_MODE = False
                else:
                    HOLIDAY_MODE = True
            else:
                HOLIDAY_MODE = True
            holdings_df = pd.read_csv(ETF_HOLDINGS_PATH)

            if HOLIDAY_MODE:
                df_stocks = None
                print("  Lideres sectoriales omitidos (df_stocks no disponible).")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  Carga de df_stocks omitida: {e}")

    return {
        "df_stocks": df_stocks,
        "df_stocks_effective_meta": df_stocks_effective_meta,
        "holdings_df": holdings_df,
    }
