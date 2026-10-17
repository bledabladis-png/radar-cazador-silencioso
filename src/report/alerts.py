# -*- coding: utf-8 -*-
"""Alertas de divergencia y Cross-Module Conflict.

Extraidos de src/report_generator.py (refactor C1, fase C1-6a2).
"""

import pandas as pd

from config.tickers import SECTOR_NAMES
from config.settings import NEUTRAL_BAND_HYG_LQD_ZSCORE


def render_alerts(breadth_values, liquidity_regime, price_flow_divergences, confirmation_data=None):
    """Renderiza la seccion Alertas de Divergencia (Inicial).

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    alerts = []
    
    if breadth_values:
        ema200 = breadth_values.get('% sobre EMA200', 0)
        ema20 = breadth_values.get('% sobre EMA20', 0)
        if ema200 > 0.70 and ema20 < 0.60:
            alerts.append(f"- **Breadth Divergence:** Breadth EMA200: {ema200:.0%}; Breadth EMA20: {ema20:.0%}. La amplitud de corto plazo es inferior a la de largo plazo.")
    
    if liquidity_regime == 'HIGH_STRESS':
        # H1 (2026-10-15, auditoria externa): el texto afirmaba
        # "desviacion positiva" sin comprobar el Z-score real. Ahora se
        # condiciona al valor publicado en Cross-Asset Ratios.
        # Fuente: confirmation_data['ratios']['hyg_lqd_zscore'].
        _hyg_z = None
        if confirmation_data:
            _ratios = confirmation_data.get('ratios', {}) or {}
            _hyg_z = _ratios.get('hyg_lqd_zscore', None)

        if _hyg_z is None or (isinstance(_hyg_z, float) and pd.isna(_hyg_z)):
            alerts.append(
                "- **Financial Stress vs Credit:** Condiciones financieras "
                "elevadas. Z-score de HYG/LQD no disponible: no puede "
                "determinarse si el estres esta localizado o si el credito "
                "lo confirma."
            )
        elif abs(_hyg_z) < NEUTRAL_BAND_HYG_LQD_ZSCORE:
            # M1 (dictamen auditor externo 2026-10-16): banda neutral
            # |Z| < 0.5. Una desviacion marginal no confirma por si sola
            # el estres crediticio. Se mantiene la alerta de condiciones
            # financieras elevadas (sus propias condiciones se cumplen),
            # pero se separa esa observacion de una supuesta confirmacion
            # crediticia. Umbral documentado como criterio de
            # clasificacion, no como constante estadistica universal.
            alerts.append(
                f"- **Financial Stress vs Credit:** Condiciones financieras "
                f"elevadas. El ratio crediticio (HYG/LQD) esta proximo a "
                f"neutral (Z={_hyg_z:+.2f}, banda neutral |Z|<{NEUTRAL_BAND_HYG_LQD_ZSCORE}): no "
                f"confirma por si solo el estres sistemico. La observacion "
                f"de condiciones financieras elevadas se mantiene por sus "
                f"propias condiciones."
            )
        elif _hyg_z > 0:
            alerts.append(
                f"- **Financial Stress vs Credit:** Condiciones financieras "
                f"elevadas, pero el credito relativo (HYG/LQD) presenta una "
                f"desviacion positiva frente a su distribucion reciente "
                f"(Z={_hyg_z:+.2f}). Estres localizado. No se observa "
                f"confirmacion suficiente de estres sistemico."
            )
        else:
            alerts.append(
                f"- **Financial Stress vs Credit:** Condiciones financieras "
                f"elevadas Y el credito relativo (HYG/LQD) muestra desviacion "
                f"negativa frente a su distribucion reciente (Z={_hyg_z:+.2f}). "
                f"El estres no esta localizado: el credito lo confirma."
            )
    
    if price_flow_divergences:
        for ticker, div in price_flow_divergences.items():
            if div.get('status') != 'ALIGNED':
                name = SECTOR_NAMES.get(ticker, ticker)
                # Fix F (2026-09-30): usar el message del detector, no
                # un literal fijo. Antes se publicaba "Precio fuerte"
                # para PRICE_WEAK_FLOW_SUPPORTIVE. Verificado con XLU.
                _msg = div.get('message') or '(sin detalle)'
                alerts.append(f"- **{name} Price-Flow:** {_msg}")
    
    if alerts:
        out.append("### Alertas de Divergencia (Inicial)\n")
        for alert in alerts:
            out.append(alert + "\n")
        out.append("\n")
    else:
        out.append("### Alertas de Divergencia (Inicial)\n")
        out.append("*Sin divergencias relevantes.*\n\n")
    return out


def render_cross_module(cross_module_conflict):
    """Renderiza la seccion Cross-Module Conflict.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if cross_module_conflict:
        level = cross_module_conflict.get('conflict_level', 'MIXED')
        icon = 'OK' if level == 'CONSENSUS' else 'WARN' if level in ('CONFLICT', 'DIVERGENCE') else 'INFO'
        out.append(f"### {icon} Cross-Module: {level}\n")
        out.append(f"**Mensaje:** {cross_module_conflict.get('message', '')}\n")
        blocks = cross_module_conflict.get('blocks', '')
        if blocks:
            out.append(f"**Bloques:** {blocks}\n")
        details = cross_module_conflict.get('details', {})
        if details:
            out.append("\n**Detalle por módulo:**\n")
            for mod_name, mod_info in details.items():
                state = mod_info.get('state', 'N/A')
                if state is None or str(state) == 'None':
                    state = 'N/A'
                bias_fin = mod_info.get('bias_financial', 0)
                bias_inf = mod_info.get('bias_inflation', 0)
                bias_str = ''
                if bias_fin == -1: bias_str += 'Estres Financiero '
                if bias_inf == -1: bias_str += 'Presión Inflacionaria '
                if bias_str == '': bias_str = 'Neutral'
                out.append(f"- {mod_name}: {state} ({bias_str})\n")
        out.append("\n")
    return out
