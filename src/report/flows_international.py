# -*- coding: utf-8 -*-
"""Flujos internacionales: BlackRock (DAXEX, ISF.L, IWM), Amundi (LYXI),
QQQ SEC, CFTC.

Extraidos de src/report_generator.py (refactor C1, fase C1-6d4a).
"""

import pandas as pd


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
