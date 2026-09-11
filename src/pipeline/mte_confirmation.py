# -*- coding: utf-8 -*-
"""Fase 10a del pipeline: MTE + Cross-Module Conflict + Confirmation Data.

Extraido de run.py (refactor C2, fase C2-10a).
"""

import pandas as pd

from datetime import datetime

from src.utils import detect_cross_module_conflict


def _compute_mte(df_market, financial_score, all_signals, pcr_data, darkpool_data):
    print("Calculando Market Transition Engine...")
    mte_result = None
    try:
        from indicators.mte import compute_mte
        fc_score = financial_score
        cred_signal = all_signals['credit'] if 'all_signals' in dir() and 'credit' in all_signals.columns else 0
        vol_signal = all_signals['volatility'] if 'all_signals' in dir() and 'volatility' in all_signals.columns else 0

        # Verificar frescura de Dark Pool antes de pasarlo al MTE
        mte_darkpool = darkpool_data
        if darkpool_data:
            week = darkpool_data.get('week', '')
            if week:
                try:
                    d = pd.Timestamp(week)
                    age = (datetime.now() - d).days
                    if age > 14:
                        print(f"    Dark Pool ARCHIVAL ({age}d). Excluido del MTE.")
                        mte_darkpool = None
                except Exception:
                    pass

        mte_result = compute_mte(df_market, fc_score, cred_signal, vol_signal, pcr_data, mte_darkpool)
        if mte_result:
            print(f"  Escenario: {mte_result['scenario']} (MSI: {mte_result['msi']:.0f}, IPI: {mte_result['ipi']:.0f})")
        else:
            print("  MTE no disponible")
    except Exception as e:
        print(f"  Modulo MTE omitido: {e}")
    return mte_result


def _compute_cross_module(macro_regime, financial_regime, vol_regime, real_liq_regime, mte_result):
    cross_module_conflict = detect_cross_module_conflict(
        macro_regime=macro_regime,
        financial_regime=financial_regime,
        volatility_regime=vol_regime,
        liquidity_regime=real_liq_regime if real_liq_regime != 'N/A' else None,
        mte_scenario=mte_result.get('scenario') if mte_result else None
    )
    if cross_module_conflict['conflict_level'] in ('CONFLICT', 'DIVERGENCE'):
        print(f"    CROSS-MODULE {cross_module_conflict['conflict_level']}: {cross_module_conflict['message']}")
    return cross_module_conflict


def _compute_confirmation(df_market, df_stocks):
    confirmation_data = {}

    # T10Y3M
    try:
        t10y3m_df = pd.read_csv('data/macro_manual/10y3m.csv', index_col=0, parse_dates=True)
        if not t10y3m_df.empty:
            confirmation_data['t10y3m'] = float(t10y3m_df['T10Y3M'].iloc[-1])
    except Exception as e:
        print(f"  [WARN] t10y3m confirmation: {e}")
        confirmation_data['t10y3m'] = None

    # Vol Metrics
    try:
        from indicators.vol_metrics import compute_vol_metrics
        vol_data = compute_vol_metrics(df_market)
        confirmation_data.update(vol_data)
    except Exception as e:
        print(f"    Vol Metrics: Error - {e}")

    # Cross-Asset Ratios con tendencia
    try:
        from indicators.cross_asset import compute_cross_asset_ratios
        ratios = compute_cross_asset_ratios(df_market)
        confirmation_data['ratios'] = ratios
    except Exception as e:
        print(f"    Cross-Asset Ratios: Error - {e}")
        confirmation_data['ratios'] = {}

    # FLS
    try:
        from indicators.fls import compute_fls
        fls_data = compute_fls()
        if fls_data:
            confirmation_data['fls'] = fls_data
            stressed = fls_data.get('stressed_components', fls_data.get('components', 0))
            total_comp = fls_data.get('total_components', 5)
            print(f"    FLS: {fls_data['fls_normalized']:.2f} ({stressed}/{total_comp} componentes en estres)")
    except Exception as e:
        print(f"    FLS: Error - {e}")

    # Advance/Decline
    try:
        from indicators.breadth_equity import compute_advance_decline
        ad_data = compute_advance_decline(df_stocks) if df_stocks is not None else None
        if ad_data:
            confirmation_data['ad'] = ad_data
            print(f"    A/D: Net={ad_data['ad_net']:+d}  NH/NL={ad_data['nh_nl']:+d}  Thrust={ad_data['breadth_thrust']:.2f}")
        else:
            confirmation_data['ad'] = None
            print("    A/D: Sin datos suficientes (cobertura temporal baja). Se omite.")
    except Exception as e:
        print(f"    A/D: Error - {e}")
        confirmation_data['ad'] = None

    if confirmation_data:
        print(f"  Institutional Confirmation: T10Y3M={confirmation_data.get('t10y3m', 'N/A')}%")

    return confirmation_data


def compute_mte_confirmation(df_market, df_stocks, financial_score, all_signals,
                              pcr_data, darkpool_data, macro_regime,
                              financial_regime, vol_regime, real_liq_regime):
    """Ejecuta MTE + Cross-Module + Confirmation.

    Returns:
        dict con keys: mte_result, cross_module_conflict, confirmation_data
    """
    mte_result = _compute_mte(df_market, financial_score, all_signals, pcr_data, darkpool_data)
    cross_module_conflict = _compute_cross_module(
        macro_regime, financial_regime, vol_regime, real_liq_regime, mte_result
    )
    confirmation_data = _compute_confirmation(df_market, df_stocks)
    return {
        'mte_result': mte_result,
        'cross_module_conflict': cross_module_conflict,
        'confirmation_data': confirmation_data,
    }
