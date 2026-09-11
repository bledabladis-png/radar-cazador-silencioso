# -*- coding: utf-8 -*-
"""Alertas de divergencia y Cross-Module Conflict.

Extraidos de src/report_generator.py (refactor C1, fase C1-6a2).
"""

from config.tickers import SECTOR_NAMES


def render_alerts(breadth_values, liquidity_regime, price_flow_divergences):
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
        alerts.append("- **Financial Stress vs Credit:** Condiciones financieras elevadas, pero el crédito relativo (HYG/LQD) presenta una desviación positiva frente a su distribucion reciente. Estres localizado. No se observa confirmación suficiente de estres sistemico.")
    
    if price_flow_divergences:
        for ticker, div in price_flow_divergences.items():
            if div.get('status') != 'ALIGNED':
                name = SECTOR_NAMES.get(ticker, ticker)
                alerts.append(f"- **{name} Price-Flow:** Precio fuerte sin confirmación del Flow Proxy. El indicador no permite inferir directamente participacion institucional.")
    
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
