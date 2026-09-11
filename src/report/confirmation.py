# -*- coding: utf-8 -*-
"""Confirmation Data (Nivel 2) con Cross-Asset Ratios.

Extraido de src/report_generator.py (refactor C1, fase C1-6d4c2).
"""

import pandas as pd


def render_confirmation(confirmation_data):
    """Renderiza la seccion Confirmation Data (Nivel 2).

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if confirmation_data:
        out.append("## Confirmation Data (Nivel 2)\n")
        out.append("> *Indicadores de confirmación. No modifican el macro_score.*\n\n")
        
        if confirmation_data.get('t10y3m') is not None:
            sign = '+' if confirmation_data['t10y3m'] >= 0 else ''
            out.append(f"- **10Y-3M Spread:** {sign}{confirmation_data['t10y3m']:.2f}%\n")
        if confirmation_data.get('rv_21d') is not None:
            rv21 = confirmation_data['rv_21d']
            rv21_str = f'{rv21*100:.2f}%' if pd.notna(rv21) else 'N/D'
            out.append(f"- **Realized Vol (21d):** {rv21_str}\n")
        if confirmation_data.get('rv_60d') is not None:
            rv60 = confirmation_data['rv_60d']
            rv60_str = f'{rv60*100:.2f}%' if pd.notna(rv60) else 'N/D'
            out.append(f"- **Realized Vol (60d):** {rv60_str}\n")
        if confirmation_data.get('vrp_21d') is not None:
            vrp21 = confirmation_data['vrp_21d']
            vrp21_str = f'{vrp21*100:+.2f}%' if pd.notna(vrp21) else 'N/D'
            out.append(f"- **VRP Proxy (VIX - RV21):** {vrp21_str}\n")
        if confirmation_data.get('vrp_60d') is not None:
            vrp60 = confirmation_data['vrp_60d']
            vrp60_str = f'{vrp60*100:+.2f}%' if pd.notna(vrp60) else 'N/D'
            out.append(f"- **VRP Proxy (VIX - RV60):** {vrp60_str}\n")

        fls = confirmation_data.get('fls', {})
        if fls:
            fls_score = fls.get('fls_normalized', 0)*100
            fls_comp = fls.get('components', 0)
            fls_total = fls.get('total_components', 5)
            stressed = fls.get('stressed_components', fls_comp)
            out.append(f"- **Funding & Liquidity Stress (FLS):** {fls_score:.0f}/100 ")
            out.append(f"({stressed}/{fls_total} componentes en estres)\n")
            fls_detail = fls.get('detail', {})
            if fls_detail:
                out.append("  - Desglose:\n")
                for comp_name, comp_val in fls_detail.items():
                    stress_mark = 'WARN' if comp_val.get('stressed', False) else 'OK'
                    val = comp_val.get('value', 0)
                    val_str = f'{val:.2f}' if val is not None else 'N/D'
                    out.append(f"    {stress_mark} {comp_name}: {val_str}\n")

        ad = confirmation_data.get('ad', {})
        if ad:
            out.append(f"- **Advance/Decline Net:** {ad.get('ad_net', 0):+d} ({ad.get('advances', 0)} avances / {ad.get('declines', 0)} descensos)\n")
            out.append(f"- **New Highs/Lows (mercado):** {ad.get('new_highs', 0)} maximos / {ad.get('new_lows', 0)} minimos (NH-NL: {ad.get('nh_nl', 0):+d})\n")
            thrust = ad.get('breadth_thrust', 0.5)
            if thrust > 0.70 or thrust < 0.30:
                out.append(f"- **Breadth Thrust extremo:** {thrust*100:.1f}%\n")
            out.append(f"- **A/D Line (acumulada):** {ad.get('ad_line', 0):.0f}\n")

        mte_scenario_conf = confirmation_data.get('mte_scenario', '')
        if mte_scenario_conf == 'RECESSION':
            nh_nl = ad.get('nh_nl', 0)
            if nh_nl < 0:
                out.append(f"- **RECESSION CAPITULATION SIGNAL:** NH/NL negativo ({nh_nl:+d}). Evidencia preliminar de posible rebote tactico.\n")

        ratios = confirmation_data.get('ratios', {})
        if ratios:
            out.append("\n### Cross-Asset Ratios\n")
            out.append("| Ratio | Valor | Delta 20d | Z-Score (60d) |\n")
            out.append("|-------|-------|-----------|---------------|\n")
            ratio_names = {
                'copper_gold': 'Copper/Gold',
                'tlt_ief': 'TLT/IEF',
                'tip_ief': 'TIP/IEF',
                'dxy_em': 'DXY/EEM',
                'hyg_lqd': 'HYG/LQD',
                'kre_spy': 'KRE/SPY',
                'sox_spy': 'SMH/SPY',
                'iyt_spy': 'IYT/SPY',
                'xle_spy': 'XLE/SPY',
                'xlu_spy': 'XLU/SPY',
                'xlv_spy': 'XLV/SPY',
                'xlp_spy': 'XLP/SPY',
            }
            for key, label in ratio_names.items():
                if key in ratios and ratios[key] is not None:
                    val = ratios[key]
                    delta_key = f'{key}_delta20'
                    z_key = f'{key}_zscore'
                    delta = ratios.get(delta_key, None)
                    z = ratios.get(z_key, None)
                    delta_str = f'{delta*100:+.1f}%' if delta is not None and pd.notna(delta) else 'N/D'
                    z_str = f'{z:+.2f}' if z is not None and pd.notna(z) else 'N/D'
                    out.append(f"| {label} | {val:.4f} | {delta_str} | {z_str} |\n")
        out.append("\n")
    return out
