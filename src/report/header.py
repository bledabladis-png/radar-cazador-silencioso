# -*- coding: utf-8 -*-
"""Seccion Header + Resumen de Regimenes del reporte diario.

Extraido de src/report_generator.py (refactor C1, fase C1-5).
"""

import pandas as pd

from src.report.helpers import _fmt_signed
from config.settings import FLS_STRESS_THRESHOLD


def render_regimenes(macro_score, macro_regime, macro_conf,
                     liquidity_score, financial_regime, liq_conf,
                     volatility_score, vol_regime, vol_conf,
                     real_liquidity_regime, real_liquidity_conf,
                     real_liq_score, real_liq_prev,
                     sector_regime, confirmation_data=None):
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
    except (AttributeError, IndexError, ValueError, TypeError):
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
    except (AttributeError, IndexError, ValueError, TypeError):
        cond_score = float('nan')
    cond_score_str = f'{cond_score:.2f}' if pd.notna(cond_score) else 'N/D'
    liq_conf_str = f'{liq_conf:.0%}' if pd.notna(liq_conf) else 'N/D'
    out.append(f"- **Cond. Financieras:** {financial_regime} (Score: {cond_score_str}, Signal Consistency: {liq_conf_str})\n")
    # C2 (2026-10-17): aviso simetrico al de macro (linea 38). Si la
    # consistencia es baja, los componentes (VIX, credito, dolar,
    # curva) no coinciden aunque el score agregado de una etiqueta.
    # La etiqueta refleja el score, no consenso entre componentes.
    if pd.notna(liq_conf) and liq_conf < 0.30:
        out.append("  *Signal Consistency baja: los componentes (VIX, credito, dolar, curva) no coinciden. La etiqueta refleja el score agregado, no consenso.*\n")
    if financial_regime == 'HIGH_STRESS':
        # H15 (2026-10-17, REP-AUDIT-20261017-capa2): modular el
        # texto segun FLS. Coherente con la critica del auditor:
        # 'NEUTRAL de liquidez FRED' no contradice 'FLS 85/100';
        # el reporte debe explicitar que son dimensiones distintas.
        _fls_score = None
        if confirmation_data:
            _fls = confirmation_data.get('fls', {}) or {}
            _fls_norm = _fls.get('fls_normalized')
            if _fls_norm is not None:
                try:
                    _fls_score = float(_fls_norm) * 100.0
                except (TypeError, ValueError):
                    _fls_score = None
        if _fls_score is not None and _fls_score >= FLS_STRESS_THRESHOLD:
            out.append(
                f"  *Nota: El modulo financiero detecta estres significativo. "
                f"El Funding & Liquidity Stress (FLS: {_fls_score:.0f}/100) "
                f"SI senala estres de corto plazo. La liquidez real (FRED) "
                f"es una dimension distinta y se reporta por separado. "
                f"No se clasifica como CRISIS sistemica.*\n"
            )
        else:
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
            out.append(f"  - *Liquidity Delta (vs ejecución anterior): {_fmt_signed(delta, '{:+.3f}', '{:.3f}')} ({delta_str})*\n")
        except (ValueError, TypeError, IndexError, AttributeError):
            pass
    
    # Fix: alinear con patron de lineas 26/40 (macro_score, liquidity_score).
    try:
        vol_z = float(volatility_score.iloc[-1]) if hasattr(volatility_score, 'iloc') else float(volatility_score)
    except (AttributeError, IndexError, ValueError, TypeError):
        vol_z = float('nan')
    if pd.notna(vol_z) and vol_conf < 0.05 and abs(vol_z) < 0.1:
        vol_conf_str = "Señal neutra (sin desviación significativa)"
    else:
        vol_conf_str = f"Signal Consistency: {vol_conf:.0%}"
    vol_z_display = "N/D" if pd.isna(vol_z) else ("0.00" if abs(vol_z) < 0.005 else f"{vol_z:.2f}")
    out.append(f"- **Volatilidad:** {vol_regime} (Z-Score: {vol_z_display}, {vol_conf_str})\n")
    # H15-vol (2026-10-17, REP-AUDIT-20261017-capa6-commitE):
    # el texto anterior decia 'VELOCIDAD de cambio (z-score del
    # retorno del VIX)', lo cual es incorrecto. compute_volatility_regime
    # NO usa el retorno del VIX: usa std(returns, window=20) normalizada
    # contra baseline de 756d (ver regimes/volatility_regime.py y
    # indicators/volatility.py::volatility_regime). Es volatilidad
    # realizada, no velocidad de cambio. Verificado con simulacion
    # 2026-10-17: retorno diario del VIX +21.45% -> z=-0.12 (porque
    # un dia no mueve la std rolling de 20 sesiones). El nivel del VIX
    # (percentil) se reporta en 'Estructura de volatilidad'.
    out.append("  *Este regimen mide la VOLATILIDAD REALIZADA del VIX: desviacion estandar rolling de sus retornos diarios (ventana 20 sesiones), normalizada contra su baseline de 3 anos. NO mide el retorno diario del VIX. El NIVEL del VIX (percentil del valor absoluto) se reporta en la tabla 'Estructura de volatilidad'. Son dimensiones distintas y pueden no coincidir.*\n")

    out.append(f"- **Sectores:** {sector_regime}\n")
    # H5 (2026-10-15) + H10/H19 (2026-10-17): Conf. Frescura se deriva
    # SOLO de la frescura. Renombrada para evitar que se interprete
    # como confianza global (el auditor del 2026-10-17 lo marco como
    # riesgo: una fuente reciente con cobertura 18% aparecia como
    # "Alta"). La cobertura es dimension independiente y se reporta
    # en la tabla Calidad.
    out.append("*Nota: Signal Consistency mide la consistencia entre señales, no una probabilidad estadistica calibrada. Conf. Frescura se deriva SOLO de la frescura observada (CURRENT/RECENT -> Alta; STALE/ARCHIVAL -> Baja), NO de la cobertura. Los umbrales de frescura por fuente estan en config/settings.py (FRESHNESS_*).*\n\n")


    return out
