# -*- coding: utf-8 -*-
"""Fase 2 del pipeline: regimenes (Financial + Liquidity + Volatility + Macro).

Extraido de run.py (refactor C2, fase C2-4).
"""

import pandas as pd

from regimes.financial_conditions import compute_financial_conditions
from regimes.liquidity import compute_liquidity_score as compute_real_liquidity
from regimes.volatility_regime import compute_volatility_regime
from regimes.macro_regime import compute_macro_regime
from src.utils import get_col


def compute_all_regimes(df_market, df_macro_manual):
    """Calcula los 4 regimenes del sistema.

    Returns:
        dict con keys:
            financial_score, financial_regime, liq_conf,
            real_liq_score, real_liq_regime, real_liq_conf, real_liq_prev,
            vol_score, vol_regime, vol_conf,
            macro_score, macro_regime, macro_conf, all_signals
    """
    print("Calculando regimen de Cond. Financieras...")
    financial_score, financial_regime, liq_conf = compute_financial_conditions(df_market)
    print(f"  Cond. Financieras: {financial_regime} (conf: {liq_conf:.0%})")

    print("Calculando liquidez real (FRED)...")
    result = compute_real_liquidity()
    if result[0] is not None:
        real_liq_score, real_liq_regime, real_liq_conf, real_liq_prev = result
        print(f"  Liquidez real: {real_liq_regime} (conf: {real_liq_conf:.0%})")
    else:
        real_liq_score, real_liq_regime, real_liq_conf, real_liq_prev = None, 'N/A', 0.0, None
        print("  Liquidez real: no disponible (sin datos FRED)")

    print("Calculando regimen de volatilidad...")
    try:
        vix_close = get_col(df_market, '^VIX', 'Close')
        vix_returns = vix_close.pct_change(fill_method=None)
    except KeyError:
        print("  ^VIX no disponible, usando volatilidad plana.")
        vix_returns = pd.Series(dtype=float)

    vol_score, vol_regime, vol_conf = compute_volatility_regime(vix_returns)
    print(f"  Volatilidad: {vol_regime} (conf: {vol_conf:.0%})")

    print("Calculando regimen macro...")
    macro_score, macro_regime, macro_conf, all_signals = compute_macro_regime(
        df_market, df_macro_manual, financial_score, vol_score
    )
    print(f"  Macro: {macro_regime} (conf: {macro_conf:.0%})")

    return {
        'financial_score': financial_score,
        'financial_regime': financial_regime,
        'liq_conf': liq_conf,
        'real_liq_score': real_liq_score,
        'real_liq_regime': real_liq_regime,
        'real_liq_conf': real_liq_conf,
        'real_liq_prev': real_liq_prev,
        'vol_score': vol_score,
        'vol_regime': vol_regime,
        'vol_conf': vol_conf,
        'macro_score': macro_score,
        'macro_regime': macro_regime,
        'macro_conf': macro_conf,
        'all_signals': all_signals,
    }
