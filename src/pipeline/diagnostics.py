# -*- coding: utf-8 -*-
"""Fase 9a del pipeline: Directional Agreement + Price-Flow Divergence
+ Shock Sensitivity.

Extraido de run.py (refactor C2, fase C2-9a).
"""

import numpy as np
import pandas as pd

from src.utils import get_col
from indicators.signal_agreement import compute_signal_agreement
from indicators.price_flow_divergence import detect_price_flow_divergence


SECTOR_ETFS = ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']


def _compute_directional_agreement(df_market, tactical_scores, structural_scores, sector_flow_rank):
    signal_agreements = {}
    signal_agreements_display = {}
    try:
        for sector_etf in SECTOR_ETFS:
            signals = {}
            signals['tactical'] = tactical_scores.get(sector_etf, 0)
            signals['structural'] = structural_scores.get(sector_etf, 0)
            try:
                close_sector = get_col(df_market, sector_etf, 'Close')
                close_spy = get_col(df_market, '^GSPC', 'Close')
                rs = close_sector / close_spy
                rs20 = rs.pct_change(20, fill_method=None).iloc[-1]
                signals['rs20'] = np.tanh(rs20 * 5) if pd.notna(rs20) else 0
            except Exception as e:
                print(f"  [WARN] rs20 signal: {e}")
                signals['rs20'] = 0
            flow_val = next((f for t, f in sector_flow_rank if t == sector_etf), 0)
            signals['flow'] = flow_val
            result = compute_signal_agreement(signals)
            signal_agreements[sector_etf] = result['agreement']
            signal_agreements_display[sector_etf] = result['display']
        print(f"    Directional Agreement calculado para {len(signal_agreements)} sectores.")
    except Exception as e:
        print(f"    Directional Agreement omitido: {e}")
        signal_agreements = {s: 0.5 for s in SECTOR_ETFS}
        signal_agreements_display = {s: '50% MIXED' for s in SECTOR_ETFS}
    return signal_agreements, signal_agreements_display


def _compute_price_flow_divergence(df_market, sector_flow_rank):
    price_flow_divergences = {}
    try:
        for sector_etf in SECTOR_ETFS:
            try:
                close_sector = get_col(df_market, sector_etf, 'Close')
                price_ret_20d = (close_sector.iloc[-1] / close_sector.iloc[-21] - 1) if len(close_sector) >= 21 else 0.0
            except Exception as e:
                print(f"  [WARN] price_ret_20d: {e}")
                price_ret_20d = 0.0
            flow_val = next((f for t, f in sector_flow_rank if t == sector_etf), 0)
            price_flow_divergences[sector_etf] = detect_price_flow_divergence(price_ret_20d, flow_val)
        for sector_etf, div in price_flow_divergences.items():
            if div['status'] != 'ALIGNED':
                name = sector_etf
                print(f"    Price-Flow Divergence [{name}]: {div['status']}")
        print(f"    Price-Flow Divergence calculado para {len(price_flow_divergences)} sectores.")
    except Exception as e:
        print(f"    Price-Flow Divergence omitido: {e}")
        price_flow_divergences = {s: {'status': 'ALIGNED', 'message': ''} for s in SECTOR_ETFS}
    return price_flow_divergences


def _compute_shock_sensitivity(df_market):
    shock_sensitivities = {}
    try:
        from indicators.commodity_market_correlation import compute_commodity_market_correlation
        for sector_etf in SECTOR_ETFS:
            shock_sensitivities[sector_etf] = compute_commodity_market_correlation(df_market, sector_etf)
        print(f"    Shock Sensitivity calculada para {len(shock_sensitivities)} sectores.")
    except Exception as e:
        print(f"    Shock Sensitivity omitida: {e}")
        shock_sensitivities = {s: {} for s in SECTOR_ETFS}
    return shock_sensitivities


def compute_diagnostics(df_market, tactical_scores, structural_scores, sector_flow_rank):
    """Ejecuta Directional Agreement + Price-Flow + Shock Sensitivity.

    Returns:
        dict con keys:
            signal_agreements, signal_agreements_display,
            price_flow_divergences, shock_sensitivities
    """
    signal_agreements, signal_agreements_display = _compute_directional_agreement(
        df_market, tactical_scores, structural_scores, sector_flow_rank
    )
    price_flow_divergences = _compute_price_flow_divergence(df_market, sector_flow_rank)
    shock_sensitivities = _compute_shock_sensitivity(df_market)
    return {
        'signal_agreements': signal_agreements,
        'signal_agreements_display': signal_agreements_display,
        'price_flow_divergences': price_flow_divergences,
        'shock_sensitivities': shock_sensitivities,
    }
