# -*- coding: utf-8 -*-
"""Flujos internacionales: BlackRock (DAXEX, ISF.L, IWM), Amundi (LYXI),
QQQ SEC, CFTC.

Extraidos de src/report_generator.py (refactor C1, fase C1-6d4a).
"""

import pandas as pd

from src.report.helpers import _fmt_num


def render_flujo_daxex(blackrock_dax_flow):
    """Renderiza la seccion DAXEX."""
    out = []
    if blackrock_dax_flow is not None and not blackrock_dax_flow.empty:
        row = blackrock_dax_flow.iloc[-1]
        out.append("## Flujo Primario DAXEX (BlackRock)\n")
        out.append(f"- **Última fecha:** {row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date']}\n")
        out.append(f"- **NAV:** {row['nav']:.4f}\n")
        out.append(f"- **Shares Outstanding:** {row['shares_outstanding']:,.0f}\n")
        out.append(f"- **Δ Shares:** {row['shares_change']:+,.0f}\n")
        out.append(f"- **Flujo Estimado (EUR):** {row['estimated_flow_eur']:+,.2f}\n")
        out.append(f"- **Flujo % AUM:** {row['flow_pct_assets']*100:+.6f}%\n")
        out.append(f"- **Flow Z-Score:** {row['flow_zscore']:+.2f}\n")
        out.append("\n*Fuente: BlackRock. ETF Primary Flow = ΔSharesOutstanding × NAV.*\n\n")
    return out

def render_flujo_isf(blackrock_isf_flow):
    """Renderiza la seccion ISF."""
    out = []
    if blackrock_isf_flow is not None and not blackrock_isf_flow.empty:
        row = blackrock_isf_flow.iloc[-1]
        out.append("## Flujo Primario ISF.L (BlackRock)\n")
        out.append(f"- **Última fecha:** {row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date']}\n")
        out.append(f"- **NAV:** {row['nav']:.4f}\n")
        out.append(f"- **Shares Outstanding:** {row['shares_outstanding']:,.0f}\n")
        out.append(f"- **Δ Shares:** {row['shares_change']:+,.0f}\n")
        out.append(f"- **Flujo Estimado (GBP):** {row['estimated_flow_eur']:+,.2f}\n")
        out.append(f"- **Flujo % AUM:** {row['flow_pct_assets']*100:+.6f}%\n")
        out.append(f"- **Flow Z-Score:** {row['flow_zscore']:+.2f}\n")
        out.append("\n*Fuente: BlackRock. ETF Primary Flow = ΔSharesOutstanding × NAV.*\n\n")
    return out

def render_flujo_lyxi(amundi_lyxi_flow):
    """Renderiza la seccion LYXI."""
    out = []
    if amundi_lyxi_flow is not None and not amundi_lyxi_flow.empty:
        row = amundi_lyxi_flow.iloc[-1]
        out.append("## Flujo Primario LYXI (Amundi)\n")
        out.append(f"- **Última fecha:** {row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date']}\n")
        out.append(f"- **Shares Outstanding:** {row['shares_outstanding']:,.0f}\n")
        out.append(f"- **NAV:** {row['nav']:.4f}\n")
        out.append(f"- **AUM:** {row['class_aum']:,.2f}\n")
        if pd.notna(row.get('shares_change')):
            out.append(f"- **Δ Shares:** {row['shares_change']:+,.0f}\n")
            out.append(f"- **Flujo Estimado (EUR):** {row['estimated_flow_eur']:+,.2f}\n")
            out.append(f"- **Flujo % AUM:** {row['flow_pct_assets']*100:+.6f}%\n")
            out.append(f"- **Flow Z-Score:** {row['flow_zscore']:+.2f}\n")
        else:
            out.append("- **Δ Shares:** N/D (histórico insuficiente)\n")
            out.append("- **Flujo Estimado:** N/D\n")
        out.append("\n*Fuente: Amundi. ETF Primary Flow = ΔSharesOutstanding × NAV.*\n\n")
    return out

def render_flujo_iwm(blackrock_iwm_flow):
    """Renderiza la seccion IWM."""
    out = []
    if blackrock_iwm_flow is not None and not blackrock_iwm_flow.empty:
        row = blackrock_iwm_flow.iloc[-1]
        out.append("## Flujo Primario IWM (BlackRock)\n")
        out.append(f"- **Última fecha:** {row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date']}\n")
        out.append(f"- **NAV:** {row['nav']:.4f}\n")
        out.append(f"- **Shares Outstanding:** {row['shares_outstanding']:,.0f}\n")
        out.append(f"- **Δ Shares:** {row['shares_change']:+,.0f}\n")
        out.append(f"- **Flujo Estimado (USD):** {row['primary_flow_usd']:+,.2f}\n")
        out.append(f"- **Flujo % AUM:** {row['primary_flow_pct']:+.6f}%\n")
        out.append(f"- **Flow Z-Score:** {row['primary_flow_z']:+.2f}\n")
        out.append("\n*Fuente: BlackRock. ETF Primary Flow = ΔSharesOutstanding × NAV.*\n\n")
    return out

def render_flujo_qqq_sec(qqq_sec_flow):
    """Renderiza la seccion QQQSEC."""
    out = []
    if qqq_sec_flow is not None and not qqq_sec_flow.empty:
        row = qqq_sec_flow.iloc[-1]
        out.append("## Flujo Primario QQQ (SEC, Trimestral/Semestral)\n")
        period = str(row.get('period_type', 'N/A')).upper()
        period_date = str(row.get('period_end_date', 'N/A'))
        out.append(f"- **Período:** {period} {period_date}\n")
        out.append(f"- **Fecha de presentación:** {row.get('filing_date', 'N/A')}\n")
        out.append(f"- **Shares sold:** {row.get('shares_sold', 0):,.0f}\n")
        out.append(f"- **Shares repurchased:** {row.get('shares_repurchased', 0):,.0f}\n")
        out.append(f"- **Net shares flow:** {row.get('net_shares_flow', 0):,.0f}\n")
        out.append(f"- **Proceeds from shares sold:** {row.get('proceeds_shares_sold', 0):,.2f}\n")
        out.append(f"- **Value of shares repurchased:** {row.get('value_shares_repurchased', 0):,.2f}\n")
        out.append(f"- **Primary flow USD (oficial):** {row.get('primary_flow_usd', 0):,.2f}\n")
        out.append("\n*Fuente: SEC EDGAR, formularios N-30B-2 / N-CSRS. Frecuencia anual/semestral. No es flujo diario.*\n\n")
    return out

def render_posicionamiento_cftc(cftc_position_flow_data):
    """Renderiza la seccion CFTC."""
    out = []
    if cftc_position_flow_data is not None and not cftc_position_flow_data.empty:
        out.append("## Posicionamiento CFTC (TFF, Semanal)\n")
        out.append("| Fecha | Contrato | Participante | Net Position | Pos Change | Flow Z |\n")
        out.append("|-------|----------|--------------|--------------|------------|--------|\n")
        for _, row in cftc_position_flow_data.iterrows():
            fecha = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
            out.append(f"| {fecha} | {row['contract']} | {row['participant']} | {row['net_position']:,.0f} | {row['position_change']:+,.0f} | {row['flow_z']:+.2f} |\n")
        out.append("\n*Fuente: CFTC Traders in Financial Futures (Futures Only). Frecuencia semanal.*\n\n")
    return out

def render_flujo_posicional_nport(nport_position_change_data):
    """Renderiza la tabla Flujo Posicional N-PORT (Trimestral).

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if nport_position_change_data is not None and not nport_position_change_data.empty:
        out.append("## Flujo Posicional N-PORT (Trimestral)\n")
        out.append("*Datos del último trimestre disponible. Fuente: SEC N-PORT.*\n")
        out.append("| Fecha | Fondo | Activo | ISIN | Balance Previo | Balance Actual | Cambio | % Cambio |\n")
        out.append("|-------|-------|--------|-------|----------------|----------------|--------|-----------|\n")
        for _, row in nport_position_change_data.iterrows():
            fecha = row['REPORT_DATE'].strftime('%Y-%m-%d') if hasattr(row['REPORT_DATE'], 'strftime') else str(row['REPORT_DATE'])
            out.append(f"| {fecha} | {row['REGISTRANT_NAME']} | {row['ISSUER_NAME']} | {row['IDENTIFIER_ISIN']} | {row['PREV_BALANCE']:,.0f} | {row['BALANCE']:,.0f} | {row['POSITION_CHANGE']:+,.0f} | {row['POSITION_CHANGE_PCT']:+.2f}% |\n")
        out.append("\n")
    else:
        out.append("## Flujo Posicional N-PORT (Trimestral)\n")
        out.append("*Sin datos N-PORT disponibles en esta ejecución.*\n\n")
    return out


def render_rendimiento_qqq(qqq_performance_data):
    """Renderiza la tabla Rendimiento QQQ (Yahoo Finance).

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if qqq_performance_data is not None and not qqq_performance_data.empty:
        out.append("## Rendimiento QQQ (Yahoo Finance)\n")
        out.append("| Medida | YTD | 1Y | 3Y | 5Y | 10Y | Desde inicio |\n")
        out.append("|--------|-----|----|----|----|-----|--------------|\n")
        for _, row in qqq_performance_data.iterrows():
            label = row.get('displayLabel', 'QQQ (Yahoo Finance)')
            out.append(f"| {label} | {row['ytd']:.2f}% | {row['y1']:.2f}% | {row['y3']:.2f}% | {row['y5']:.2f}% | {row['y10']:.2f}% | {row['inception']:.2f}% |\n")
        try:
            as_of = qqq_performance_data.iloc[0].get("as_of_date", "")
            if as_of:
                out.append(f"*Fecha de cálculo (as_of_date): {as_of}*\n")
        except Exception:
            pass
        out.append("\n*Fuente: Yahoo Finance. Rendimientos calculados desde precios ajustados.*\n\n")
    # FLUJO DE PARTICIPACIONES QQQ (NPORT-P)
    return out


def render_qqq_nport_flow(qqq_nport_flow_data):
    """Renderiza la tabla Flujo de Participaciones QQQ (NPORT-P).

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if qqq_nport_flow_data is not None and not qqq_nport_flow_data.empty:
        out.append("## Flujo de Participaciones QQQ (NPORT-P)\n")
        out.append("*Fuente: SEC NPORT-P Item B.6. Frecuencia trimestral.*\n")
        try:
            report_date_str = str(qqq_nport_flow_data.iloc[0].get('report_date', 'N/A'))
            if report_date_str != 'N/A' and len(report_date_str) >= 7:
                year_month = pd.Timestamp(report_date_str)
                quarter = (year_month.month - 1) // 3 + 1
                out.append(f"*Trimestre: Q{quarter} {year_month.year}*\n")
        except Exception:
            pass
        out.append("| Mes | Ventas (M$) | Redenciones (M$) | Flujo Neto (M$) |\n")
        out.append("|-----|-------------|------------------|-----------------|\n")
        for _, row in qqq_nport_flow_data.iterrows():
            out.append(f"| {int(row['month'])} | {row['sales']/1e6:,.2f} | {row['redemptions']/1e6:,.2f} | {row['net_flow']/1e6:,.2f} |\n")
        out.append("\n")
    else:
        out.append("## Flujo de Participaciones QQQ (NPORT-P)\n")
        out.append("*Sin datos NPORT-P de QQQ en esta ejecución.*\n\n")
    return out


def render_flujo_sintesis(flow_synthesis):
    """Renderiza la tabla Flujo - Sintesis Descriptiva.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if flow_synthesis:
        out.append("## Flujo - Sintesis Descriptiva\n")
        out.append("| Capa | Lectura |\n")
        out.append("|------|---------|\n")
        out.append(f"| Flow Proxy | {_fmt_num(flow_synthesis.get('flow_proxy_sign'), '{:+.2f}')} |\n")
        out.append(f"| ETF Primary Flow | {_fmt_num(flow_synthesis.get('etf_primary_flow_sign'), '{:+.2f}')} |\n")
        out.append(f"| CFTC Position Flow | {_fmt_num(flow_synthesis.get('cftc_flow_sign'), '{:+.2f}')} |\n")
        out.append(f"| Europa Primary Flow | {flow_synthesis.get('european_flow_sign', 0):+.2f} |\n")
        out.append(f"\n**FLOW_CONFIDENCE:** {flow_synthesis.get('confidence', 'N/A')}\n")
        out.append("\n*Interpretación descriptiva: concordancia de signos entre capas. No es señal predictiva.*\n\n")
    return out
