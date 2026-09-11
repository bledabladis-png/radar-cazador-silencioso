# -*- coding: utf-8 -*-
"""Seccion Header + Resumen de Regimenes del reporte diario.

Extraido de src/report_generator.py (refactor C1, fase C1-5).
"""

import pandas as pd


def render_regimenes(macro_score, macro_regime, macro_conf,
                     liquidity_score, liquidity_regime, liq_conf,
                     volatility_score, vol_regime, vol_conf,
                     real_liquidity_regime, real_liquidity_conf,
                     real_liq_score, real_liq_prev,
                     sector_regime):
    """Renderiza la seccion 'Resumen de Regimenes'.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    # =========================================================================
    # RESUMEN DE REGIMENES
    # =========================================================================
    out.append("## Resumen de Regimenes\n")
    try:
        score_value = float(macro_score.iloc[-1])
    except Exception:
        score_value = float('nan')

    if pd.isna(score_value):
        score_str = "N/D"
    else:
        score_str = f"{score_value:.2f}"

    out.append(f"- **Macro:** {macro_regime} (Score: {score_str}, Signal Consistency: {macro_conf:.0%})\n")
    if macro_conf < 0.30:
        out.append("  *Signal Consistency baja: señales contradictorias en el entorno actual.*\n")
    
    try:
        cond_score = float(liquidity_score.iloc[-1])
    except Exception:
        cond_score = float('nan')
    cond_score_str = f'{cond_score:.2f}' if pd.notna(cond_score) else 'N/D'
    liq_conf_str = f'{liq_conf:.0%}' if pd.notna(liq_conf) else 'N/D'
    out.append(f"- **Cond. Financieras:** {liquidity_regime} (Score: {cond_score_str}, Signal Consistency: {liq_conf_str})\n")
    if liquidity_regime == 'HIGH_STRESS':
        out.append("  *Nota: El módulo financiero detecta estres significativo, pero volatilidad y liquidez no confirman un deterioro transversal. No se clasifica como CRISIS sistemica.*\n")
    
    if real_liquidity_regime is not None:
        out.append(f"- **Liquidez Real (FRED):** {real_liquidity_regime} (Signal Consistency: {real_liquidity_conf:.0%})\n")
    if real_liq_prev is not None:
        try:
            delta = float(real_liq_score.iloc[-1]) - float(real_liq_prev)
            if delta > 0.05:
                delta_str = "MEJORA"
            elif delta < -0.05:
                delta_str = "EMPEORA"
            else:
                delta_str = "ESTABLE"
            out.append(f"  - *Liquidity Delta (vs ejecución anterior): {delta:+.3f} ({delta_str})*\n")
        except:
            pass
    
    vol_z = volatility_score.iloc[-1] if hasattr(volatility_score, 'iloc') else volatility_score
    if vol_conf < 0.05 and abs(vol_z) < 0.1:
        vol_conf_str = "Señal neutra (sin desviación significativa)"
    else:
        vol_conf_str = f"Signal Consistency: {vol_conf:.0%}"
    vol_z_display = "0.00" if abs(vol_z) < 0.005 else f"{vol_z:.2f}"
    out.append(f"- **Volatilidad:** {vol_regime} (Z-Score: {vol_z_display}, {vol_conf_str})\n")

    out.append(f"- **Sectores:** {sector_regime}\n")
    out.append("*Nota: Signal Consistency mide la consistencia entre señales, no una probabilidad estadistica calibrada. Data Conf mide la frescura y cobertura de los datos.*\n\n")


    return out
