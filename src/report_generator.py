import pandas as pd
import os
from datetime import datetime
from config.tickers import SECTOR_NAMES
from config.index_tickers import INDEX_CONFIG
from src.report.alerts import render_alerts, render_cross_module
from src.report.breadth import render_breadth_market
from src.report.freshness import render_data_freshness
from src.report.sectorial import (
    render_sector_breadth,
    render_sector_concentration,
    render_sector_dispersion,
)
from src.report.leaders import (
    render_momentum_sectores,
    render_tactical_leaders,
    render_momentum_otros,
    render_structural_ranking,
    render_acciones_seleccionadas,
)
from src.report.rankings import (
    render_rankings_sectoriales,
    render_persistencia,
    render_opportunity_map,
)
from src.report.etf_flows import (
    render_flujo_spdr,
    render_flujo_caracteristicas,
    render_divergencia_precio_flujo,
)
from src.report.market_context import (
    render_liderazgo_interno,
    render_rotacion_reciente,
    render_dispersion_sectores,
    render_correlacion_sectores,
    render_contexto_cross_asset,
)
from src.report.sector_context import (
    render_matriz_regimen,
    render_representatividad_lider,
    render_wyckoff_sectorial,
    render_divergencia_sector_lideres,
    render_momentum_amplitud,
)
from src.report.sentiment import render_sentimiento_opciones
from src.report.slpm import render_slpm_v12, render_slpm_legacy
from src.report.header import render_regimenes
from src.report.helpers import (
    _fmt_num,
    _classify_finra_freshness,
)

MODEL_VERSION = "4.3"
WEIGHTS_VERSION = "3"
INDICATORS_VERSION = "2"


def generate_daily_report(macro_score, macro_regime, macro_conf, liquidity_score, liquidity_regime, liq_conf,
                          volatility_score, vol_regime, vol_conf, sector_results,
                          sector_price_rank, sector_flow_rank, otros_price_rank, otros_flow_rank,
                          leader_lines=None, breadth_values=None, real_liquidity_regime=None, real_liquidity_conf=None,
                          pcr_data=None, darkpool_data=None, mte_result=None, confirmation_data=None, slpm_data=None,
                          slpm_v12_data=None, tactical_scores=None, structural_scores=None,
                          sector_persistence=None, signal_agreements=None, signal_agreements_display=None,
                          cross_module_conflict=None, shock_sensitivities=None, price_flow_divergences=None,
                          dc_summary="", all_signals=None, real_liq_score=None, real_liq_prev=None, index_leaders=None, index_phases=None, etf_primary_flow_data=None, cftc_position_flow_data=None, flow_synthesis=None, blackrock_dax_flow=None, blackrock_isf_flow=None, amundi_lyxi_flow=None, blackrock_iwm_flow=None, nport_position_change_data=None, qqq_performance_data=None, qqq_nport_flow_data=None, qqq_sec_flow=None, sector_breadth_data=None, sector_concentration_data=None, sector_flow_characteristics_data=None, rs_internal_data=None, sector_rank_deltas_data=None, sector_regime_matrix_data=None, leader_representativeness_data=None, sector_wyckoff_distribution_data=None, sector_leader_divergence_data=None, sector_breadth_momentum_data=None, evidence_matrix_data=None, sector_dispersion_data=None, sector_correlation_summary_data=None, sector_correlation_matrix_data=None, cross_asset_context_data=None, volatility_structure_data=None, data_quality_data=None, output_path='outputs/report/reporte_diario.md'):
    lines = []
    lines.append("# MACRO SECTORIAL - Reporte Diario\n")
    lines.append(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Modelo:** v{MODEL_VERSION} | Pesos: v{WEIGHTS_VERSION} | Indicadores: v{INDICATORS_VERSION}\n\n")

    # =========================================================================
    # RESUMEN DE REGIMENES (extraido a src/report/header.py, C1-5)
    # =========================================================================
    sector_regime = sector_results['regime']
    lines.extend(render_regimenes(
        macro_score, macro_regime, macro_conf,
        liquidity_score, liquidity_regime, liq_conf,
        volatility_score, vol_regime, vol_conf,
        real_liquidity_regime, real_liquidity_conf,
        real_liq_score, real_liq_prev,
        sector_regime=sector_results['regime'],
    ))
    # =========================================================================
    # DATA FRESHNESS
    # =========================================================================
    lines.extend(render_data_freshness(pcr_data, darkpool_data, sector_results))


    # =========================================================================
    # ALERTAS DE DIVERGENCIA
    # =========================================================================
    lines.extend(render_alerts(breadth_values, liquidity_regime, price_flow_divergences))

    # =========================================================================
    # CROSS-MODULE CONFLICT
    # =========================================================================
    lines.extend(render_cross_module(cross_module_conflict))

    # =========================================================================
    # BREADTH DE MERCADO (11 sectores)
    # =========================================================================
    lines.extend(render_breadth_market(breadth_values))

    # =========================================================================
    # =========================================================================
    # SECTOR BREADTH & HEALTH
    # =========================================================================
    lines.extend(render_sector_breadth(sector_breadth_data))

    # =========================================================================
    # SECTOR CONCENTRATION
    # =========================================================================
    lines.extend(render_sector_concentration(sector_concentration_data))

    # =========================================================================
    # DISPERSIÓN INTERNA
    # =========================================================================
    lines.extend(render_sector_dispersion(sector_concentration_data))

    # =========================================================================
    # TACTICAL LEADERS
    # =========================================================================
    lines.extend(render_momentum_sectores(sector_price_rank, sector_flow_rank))
    lines.extend(render_tactical_leaders(
        tactical_scores, structural_scores,
        sector_price_rank, sector_flow_rank, shock_sensitivities,
    ))

    # =========================================================================
    # MOMENTUM OTROS ACTIVOS + STRUCTURAL RANKING
    # =========================================================================
    lines.extend(render_momentum_otros(otros_price_rank, otros_flow_rank))
    lines.extend(render_structural_ranking(
        structural_scores, tactical_scores,
        sector_persistence, signal_agreements, signal_agreements_display,
    ))

    # =========================================================================
    # RANKINGS SECTORIALES
    # =========================================================================
    lines.extend(render_rankings_sectoriales(
        sector_results, tactical_scores, structural_scores,
        sector_persistence, signal_agreements, signal_agreements_display,
        shock_sensitivities,
    ))

    # =========================================================================
    # PERSISTENCIA SECTORIAL
    # =========================================================================
    lines.extend(render_persistencia(sector_persistence))

    # =========================================================================
    # OPPORTUNITY MAP
    # =========================================================================
    lines.extend(render_opportunity_map(
        tactical_scores, structural_scores,
        sector_persistence, signal_agreements,
    ))

    # =========================================================================
    # SLPM v1.2
    # =========================================================================
    # =========================================================================
    # SLPM v1.2
    # =========================================================================
    lines.extend(render_slpm_v12(slpm_v12_data))

    # =========================================================================
    # LEGACY SLPM v1.0
    # =========================================================================
    lines.extend(render_slpm_legacy(slpm_data))

    # =========================================================================
    # ACCIONES SELECCIONADAS
    # =========================================================================
    lines.extend(render_acciones_seleccionadas(leader_lines))

    # =========================================================================
    # SENTIMIENTO DE OPCIONES (OMS v2.0)
    # =========================================================================
    lines.extend(render_sentimiento_opciones(pcr_data))

    # =========================================================================
    # ETF PRIMARY FLOW (SPDR) + CARACTERISTICAS + DIVERGENCIA
    # =========================================================================
    lines.extend(render_flujo_spdr(etf_primary_flow_data))
    lines.extend(render_flujo_caracteristicas(sector_flow_characteristics_data))
    lines.extend(render_divergencia_precio_flujo(sector_flow_characteristics_data))

    # =========================================================================
    # CONTEXTO DE MERCADO (liderazgo, rotacion, dispersion, correlacion, cross-asset)
    # =========================================================================
    lines.extend(render_liderazgo_interno(rs_internal_data))
    lines.extend(render_rotacion_reciente(sector_rank_deltas_data))
    lines.extend(render_dispersion_sectores(sector_dispersion_data))
    lines.extend(render_correlacion_sectores(sector_correlation_summary_data))
    lines.extend(render_contexto_cross_asset(cross_asset_context_data))
    # =========================================================================
    # CONTEXTO SECTORIAL (matriz, representatividad, wyckoff, divergencia, amplitud)
    # =========================================================================
    lines.extend(render_matriz_regimen(sector_regime_matrix_data))
    lines.extend(render_representatividad_lider(leader_representativeness_data))
    lines.extend(render_wyckoff_sectorial(sector_wyckoff_distribution_data))
    lines.extend(render_divergencia_sector_lideres(sector_leader_divergence_data))
    lines.extend(render_momentum_amplitud(sector_breadth_momentum_data))

    # FLUJO PRIMARIO DAXEX (BlackRock)
    # =========================================================================
    if blackrock_dax_flow is not None and not blackrock_dax_flow.empty:
        row = blackrock_dax_flow.iloc[-1]
        lines.append("## Flujo Primario DAXEX (BlackRock)\n")
        lines.append(f"- **Última fecha:** {row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date']}\n")
        lines.append(f"- **NAV:** {row['nav']:.4f}\n")
        lines.append(f"- **Shares Outstanding:** {row['shares_outstanding']:,.0f}\n")
        lines.append(f"- **Δ Shares:** {row['shares_change']:+,.0f}\n")
        lines.append(f"- **Flujo Estimado (EUR):** {row['estimated_flow_eur']:+,.2f}\n")
        lines.append(f"- **Flujo % AUM:** {row['flow_pct_assets']*100:+.6f}%\n")
        lines.append(f"- **Flow Z-Score:** {row['flow_zscore']:+.2f}\n")
        lines.append("\n*Fuente: BlackRock. ETF Primary Flow = ΔSharesOutstanding × NAV.*\n\n")

    # =========================================================================
    # FLUJO PRIMARIO ISF.L (BlackRock)
    # =========================================================================
    if blackrock_isf_flow is not None and not blackrock_isf_flow.empty:
        row = blackrock_isf_flow.iloc[-1]
        lines.append("## Flujo Primario ISF.L (BlackRock)\n")
        lines.append(f"- **Última fecha:** {row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date']}\n")
        lines.append(f"- **NAV:** {row['nav']:.4f}\n")
        lines.append(f"- **Shares Outstanding:** {row['shares_outstanding']:,.0f}\n")
        lines.append(f"- **Δ Shares:** {row['shares_change']:+,.0f}\n")
        lines.append(f"- **Flujo Estimado (GBP):** {row['estimated_flow_eur']:+,.2f}\n")
        lines.append(f"- **Flujo % AUM:** {row['flow_pct_assets']*100:+.6f}%\n")
        lines.append(f"- **Flow Z-Score:** {row['flow_zscore']:+.2f}\n")
        lines.append("\n*Fuente: BlackRock. ETF Primary Flow = ΔSharesOutstanding × NAV.*\n\n")

    # =========================================================================
    # FLUJO PRIMARIO LYXI (Amundi)
    # =========================================================================
    if amundi_lyxi_flow is not None and not amundi_lyxi_flow.empty:
        row = amundi_lyxi_flow.iloc[-1]
        lines.append("## Flujo Primario LYXI (Amundi)\n")
        lines.append(f"- **Última fecha:** {row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date']}\n")
        lines.append(f"- **Shares Outstanding:** {row['shares_outstanding']:,.0f}\n")
        lines.append(f"- **NAV:** {row['nav']:.4f}\n")
        lines.append(f"- **AUM:** {row['class_aum']:,.2f}\n")
        if pd.notna(row.get('shares_change')):
            lines.append(f"- **Δ Shares:** {row['shares_change']:+,.0f}\n")
            lines.append(f"- **Flujo Estimado (EUR):** {row['estimated_flow_eur']:+,.2f}\n")
            lines.append(f"- **Flujo % AUM:** {row['flow_pct_assets']*100:+.6f}%\n")
            lines.append(f"- **Flow Z-Score:** {row['flow_zscore']:+.2f}\n")
        else:
            lines.append("- **Δ Shares:** N/D (histórico insuficiente)\n")
            lines.append("- **Flujo Estimado:** N/D\n")
        lines.append("\n*Fuente: Amundi. ETF Primary Flow = ΔSharesOutstanding × NAV.*\n\n")

    # =========================================================================
    # FLUJO PRIMARIO IWM (BlackRock)
    # =========================================================================
    if blackrock_iwm_flow is not None and not blackrock_iwm_flow.empty:
        row = blackrock_iwm_flow.iloc[-1]
        lines.append("## Flujo Primario IWM (BlackRock)\n")
        lines.append(f"- **Última fecha:** {row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date']}\n")
        lines.append(f"- **NAV:** {row['nav']:.4f}\n")
        lines.append(f"- **Shares Outstanding:** {row['shares_outstanding']:,.0f}\n")
        lines.append(f"- **Δ Shares:** {row['shares_change']:+,.0f}\n")
        lines.append(f"- **Flujo Estimado (USD):** {row['primary_flow_usd']:+,.2f}\n")
        lines.append(f"- **Flujo % AUM:** {row['primary_flow_pct']:+.6f}%\n")
        lines.append(f"- **Flow Z-Score:** {row['primary_flow_z']:+.2f}\n")
        lines.append("\n*Fuente: BlackRock. ETF Primary Flow = ΔSharesOutstanding × NAV.*\n\n")

    # =========================================================================
    # FLUJO PRIMARIO QQQ (SEC, Trimestral/Semestral)
    # =========================================================================
    if qqq_sec_flow is not None and not qqq_sec_flow.empty:
        row = qqq_sec_flow.iloc[-1]
        lines.append("## Flujo Primario QQQ (SEC, Trimestral/Semestral)\n")
        period = str(row.get('period_type', 'N/A')).upper()
        period_date = str(row.get('period_end_date', 'N/A'))
        lines.append(f"- **Período:** {period} {period_date}\n")
        lines.append(f"- **Fecha de presentación:** {row.get('filing_date', 'N/A')}\n")
        lines.append(f"- **Shares sold:** {row.get('shares_sold', 0):,.0f}\n")
        lines.append(f"- **Shares repurchased:** {row.get('shares_repurchased', 0):,.0f}\n")
        lines.append(f"- **Net shares flow:** {row.get('net_shares_flow', 0):,.0f}\n")
        lines.append(f"- **Proceeds from shares sold:** {row.get('proceeds_shares_sold', 0):,.2f}\n")
        lines.append(f"- **Value of shares repurchased:** {row.get('value_shares_repurchased', 0):,.2f}\n")
        lines.append(f"- **Primary flow USD (oficial):** {row.get('primary_flow_usd', 0):,.2f}\n")
        lines.append("\n*Fuente: SEC EDGAR, formularios N-30B-2 / N-CSRS. Frecuencia anual/semestral. No es flujo diario.*\n\n")

    # =========================================================================
    # CFTC POSITION FLOW (TFF, Semanal)
    # =========================================================================
    if cftc_position_flow_data is not None and not cftc_position_flow_data.empty:
        lines.append("## Posicionamiento CFTC (TFF, Semanal)\n")
        lines.append("| Fecha | Contrato | Participante | Net Position | Pos Change | Flow Z |\n")
        lines.append("|-------|----------|--------------|--------------|------------|--------|\n")
        for _, row in cftc_position_flow_data.iterrows():
            fecha = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
            lines.append(f"| {fecha} | {row['contract']} | {row['participant']} | {row['net_position']:,.0f} | {row['position_change']:+,.0f} | {row['flow_z']:+.2f} |\n")
        lines.append("\n*Fuente: CFTC Traders in Financial Futures (Futures Only). Frecuencia semanal.*\n\n")

    # =========================================================================
    # FLUJO POSICIONAL N-PORT (Trimestral)
    # =========================================================================
    if nport_position_change_data is not None and not nport_position_change_data.empty:
        lines.append("## Flujo Posicional N-PORT (Trimestral)\n")
        lines.append("*Datos del último trimestre disponible. Fuente: SEC N-PORT.*\n")
        lines.append("| Fecha | Fondo | Activo | ISIN | Balance Previo | Balance Actual | Cambio | % Cambio |\n")
        lines.append("|-------|-------|--------|-------|----------------|----------------|--------|-----------|\n")
        for _, row in nport_position_change_data.iterrows():
            fecha = row['REPORT_DATE'].strftime('%Y-%m-%d') if hasattr(row['REPORT_DATE'], 'strftime') else str(row['REPORT_DATE'])
            lines.append(f"| {fecha} | {row['REGISTRANT_NAME']} | {row['ISSUER_NAME']} | {row['IDENTIFIER_ISIN']} | {row['PREV_BALANCE']:,.0f} | {row['BALANCE']:,.0f} | {row['POSITION_CHANGE']:+,.0f} | {row['POSITION_CHANGE_PCT']:+.2f}% |\n")
        lines.append("\n")
    else:
        lines.append("## Flujo Posicional N-PORT (Trimestral)\n")
        lines.append("*Sin datos N-PORT disponibles en esta ejecución.*\n\n")

    # =========================================================================
    # =========================================================================
    # RENDIMIENTO QQQ (Yahoo Finance)
    # =========================================================================
    if qqq_performance_data is not None and not qqq_performance_data.empty:
        lines.append("## Rendimiento QQQ (Yahoo Finance)\n")
        lines.append("| Medida | YTD | 1Y | 3Y | 5Y | 10Y | Desde inicio |\n")
        lines.append("|--------|-----|----|----|----|-----|--------------|\n")
        for _, row in qqq_performance_data.iterrows():
            label = row.get('displayLabel', 'QQQ (Yahoo Finance)')
            lines.append(f"| {label} | {row['ytd']:.2f}% | {row['y1']:.2f}% | {row['y3']:.2f}% | {row['y5']:.2f}% | {row['y10']:.2f}% | {row['inception']:.2f}% |\n")
        try:
            as_of = qqq_performance_data.iloc[0].get("as_of_date", "")
            if as_of:
                lines.append(f"*Fecha de cálculo (as_of_date): {as_of}*\n")
        except Exception:
            pass
        lines.append("\n*Fuente: Yahoo Finance. Rendimientos calculados desde precios ajustados.*\n\n")
    # FLUJO DE PARTICIPACIONES QQQ (NPORT-P)
    # =========================================================================
    if qqq_nport_flow_data is not None and not qqq_nport_flow_data.empty:
        lines.append("## Flujo de Participaciones QQQ (NPORT-P)\n")
        lines.append("*Fuente: SEC NPORT-P Item B.6. Frecuencia trimestral.*\n")
        try:
            report_date_str = str(qqq_nport_flow_data.iloc[0].get('report_date', 'N/A'))
            if report_date_str != 'N/A' and len(report_date_str) >= 7:
                year_month = pd.Timestamp(report_date_str)
                quarter = (year_month.month - 1) // 3 + 1
                lines.append(f"*Trimestre: Q{quarter} {year_month.year}*\n")
        except Exception:
            pass
        lines.append("| Mes | Ventas (M$) | Redenciones (M$) | Flujo Neto (M$) |\n")
        lines.append("|-----|-------------|------------------|-----------------|\n")
        for _, row in qqq_nport_flow_data.iterrows():
            lines.append(f"| {int(row['month'])} | {row['sales']/1e6:,.2f} | {row['redemptions']/1e6:,.2f} | {row['net_flow']/1e6:,.2f} |\n")
        lines.append("\n")
    else:
        lines.append("## Flujo de Participaciones QQQ (NPORT-P)\n")
        lines.append("*Sin datos NPORT-P de QQQ en esta ejecución.*\n\n")

    # =========================================================================
    # FLUJO - SINTESIS DESCRIPTIVA
    # =========================================================================
    if flow_synthesis:
        lines.append("## Flujo - Sintesis Descriptiva\n")
        lines.append("| Capa | Lectura |\n")
        lines.append("|------|---------|\n")
        lines.append(f"| Flow Proxy | {_fmt_num(flow_synthesis.get('flow_proxy_sign'), '{:+.2f}')} |\n")
        lines.append(f"| ETF Primary Flow | {_fmt_num(flow_synthesis.get('etf_primary_flow_sign'), '{:+.2f}')} |\n")
        lines.append(f"| CFTC Position Flow | {_fmt_num(flow_synthesis.get('cftc_flow_sign'), '{:+.2f}')} |\n")
        lines.append(f"| Europa Primary Flow | {flow_synthesis.get('european_flow_sign', 0):+.2f} |\n")
        lines.append(f"\n**FLOW_CONFIDENCE:** {flow_synthesis.get('confidence', 'N/A')}\n")
        lines.append("\n*Interpretación descriptiva: concordancia de signos entre capas. No es señal predictiva.*\n\n")

    # =========================================================================
    # MTE v1.0
    # =========================================================================
    # ESTRUCTURA DE VOLATILIDAD
    if volatility_structure_data is not None and not volatility_structure_data.empty:
        lines.append("## Estructura de volatilidad\n")
        lines.append("| Fecha | VIX | Perc 20d | Perc 60d | VIX3M/VIX | PCR z | Perc PCR | Lectura volatilidad | Term structure |\n")
        lines.append("|-------|-----|----------|----------|-----------|-------|----------|---------------------|----------------|\n")
        for _, row in volatility_structure_data.iterrows():
            date_str = pd.Timestamp(row['date']).strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/D'
            lines.append(f"| {date_str} | {row['vix_level']:.2f} | {row['vix_percentile_20d']:.2f} | {row['vix_percentile_60d']:.2f} | {row['term_structure_ratio']:.2f} | {row['pcr_zscore']:.2f} | {row['pcr_percentile_20d']:.2f} | {row['volatility_reading']} | {row['term_structure_reading']} |\n")
        lines.append("\n")
        lines.append("*Estructura descriptiva de volatilidad implícita y posicionamiento en opciones. No incluye Dark Pool.*\n\n")

    # CALIDAD DE DATOS
    if data_quality_data is not None and not data_quality_data.empty:
        lines.append("## Calidad, frescura y cobertura de datos\n")
        lines.append("| Fuente | Último dato | Edad (días) | Frecuencia | Frescura | Cobertura | Notas |\n")
        lines.append("|--------|-------------|--------------|------------|----------|-----------|-------|\n")
        # Mostrar solo la fila más reciente por fuente para evitar duplicados históricos
        dq = data_quality_data.copy()
        if 'date' in dq.columns and 'source' in dq.columns:
            dq['date'] = pd.to_datetime(dq['date'], errors='coerce')
            latest_idx = dq.groupby('source')['date'].idxmax()
            dq = dq.loc[latest_idx].sort_values('source')
        for _, row in dq.iterrows():
            last = row['last_date'] if pd.notna(row['last_date']) else 'N/D'
            age = f"{row['age_calendar_days']:.0f}" if pd.notna(row['age_calendar_days']) else 'N/D'
            freq = row['frequency'] if pd.notna(row['frequency']) else 'N/D'
            fresh = row['freshness'] if pd.notna(row['freshness']) else 'N/D'
            cov = f"{row['coverage']:.2%}" if pd.notna(row['coverage']) else 'N/D'
            notes = row['notes'] if pd.notna(row['notes']) else ''
            lines.append(f"| {row['source']} | {last} | {age} | {freq} | {fresh} | {cov} | {notes} |\n")
        lines.append("\n")
        lines.append("*No todas las variables tienen la misma actualidad. Los datos se muestran sin interpolación.*\n\n")
    if mte_result:
        lines.append("## Market Transition Engine (MTE v1.0)\n")
        mte_conf = mte_result.get('confidence', 0)
        mte_conf_str = f'{mte_conf:.2f}' if pd.notna(mte_conf) else 'N/D'
        mte_scenario = mte_result.get('scenario', 'N/A')
        if mte_conf < 0.5:
            lines.append(f"- **Escenario (UNCONFIRMED):** {mte_scenario} (Confidence Score no calibrado: {mte_conf_str}) - *No se considera confirmado.*\n")
        else:
            lines.append(f"- **Escenario:** {mte_scenario} (Confidence Score no calibrado: {mte_conf_str})\n")
        lines.append("*Nota: Confidence Score (no calibrado, escala 0-1) representa la distancia a los umbrales y el consenso entre motores. No debe interpretarse como probabilidad.*\n")
        lines.append(f"- **Market Stress Index (MSI):** {mte_result.get('msi', 0):.0f}\n")
        lines.append(f"- **Inflation Pressure Index (IPI):** {mte_result.get('ipi', 0):.0f}\n")
        val_srs = mte_result.get('srs', 0)
        val_srs_str = f'{val_srs:.2f}' if pd.notna(val_srs) else 'N/D'
        lines.append(f"- **Sector Rotation Score:** {val_srs_str}\n")
        val_shs = mte_result.get('shs', 0)
        val_shs_str = f'{val_shs:.2f}' if pd.notna(val_shs) else 'N/D'
        lines.append(f"- **Safe Haven Score:** {val_shs_str}\n")
        lines.append(f"- **Credit Stress Score:** {mte_result.get('cls', 0):.2f}")
        lines.append(" (orientacion: positivo = mayor estres crediticio)\n")
        ips_val = mte_result.get('ips', 0)
        ips_str = f'{ips_val:.2f}' if pd.notna(ips_val) else 'N/D'
        lines.append(f"- **Inflation Pressure Score:** {ips_str}\n\n")

    # =========================================================================
    # CONFIRMATION DATA
    # =========================================================================
    if confirmation_data:
        lines.append("## Confirmation Data (Nivel 2)\n")
        lines.append("> *Indicadores de confirmación. No modifican el macro_score.*\n\n")
        
        if confirmation_data.get('t10y3m') is not None:
            sign = '+' if confirmation_data['t10y3m'] >= 0 else ''
            lines.append(f"- **10Y-3M Spread:** {sign}{confirmation_data['t10y3m']:.2f}%\n")
        if confirmation_data.get('rv_21d') is not None:
            rv21 = confirmation_data['rv_21d']
            rv21_str = f'{rv21*100:.2f}%' if pd.notna(rv21) else 'N/D'
            lines.append(f"- **Realized Vol (21d):** {rv21_str}\n")
        if confirmation_data.get('rv_60d') is not None:
            rv60 = confirmation_data['rv_60d']
            rv60_str = f'{rv60*100:.2f}%' if pd.notna(rv60) else 'N/D'
            lines.append(f"- **Realized Vol (60d):** {rv60_str}\n")
        if confirmation_data.get('vrp_21d') is not None:
            vrp21 = confirmation_data['vrp_21d']
            vrp21_str = f'{vrp21*100:+.2f}%' if pd.notna(vrp21) else 'N/D'
            lines.append(f"- **VRP Proxy (VIX - RV21):** {vrp21_str}\n")
        if confirmation_data.get('vrp_60d') is not None:
            vrp60 = confirmation_data['vrp_60d']
            vrp60_str = f'{vrp60*100:+.2f}%' if pd.notna(vrp60) else 'N/D'
            lines.append(f"- **VRP Proxy (VIX - RV60):** {vrp60_str}\n")

        fls = confirmation_data.get('fls', {})
        if fls:
            fls_score = fls.get('fls_normalized', 0)*100
            fls_comp = fls.get('components', 0)
            fls_total = fls.get('total_components', 5)
            stressed = fls.get('stressed_components', fls_comp)
            lines.append(f"- **Funding & Liquidity Stress (FLS):** {fls_score:.0f}/100 ")
            lines.append(f"({stressed}/{fls_total} componentes en estres)\n")
            fls_detail = fls.get('detail', {})
            if fls_detail:
                lines.append("  - Desglose:\n")
                for comp_name, comp_val in fls_detail.items():
                    stress_mark = 'WARN' if comp_val.get('stressed', False) else 'OK'
                    val = comp_val.get('value', 0)
                    val_str = f'{val:.2f}' if val is not None else 'N/D'
                    lines.append(f"    {stress_mark} {comp_name}: {val_str}\n")

        ad = confirmation_data.get('ad', {})
        if ad:
            lines.append(f"- **Advance/Decline Net:** {ad.get('ad_net', 0):+d} ({ad.get('advances', 0)} avances / {ad.get('declines', 0)} descensos)\n")
            lines.append(f"- **New Highs/Lows (mercado):** {ad.get('new_highs', 0)} maximos / {ad.get('new_lows', 0)} minimos (NH-NL: {ad.get('nh_nl', 0):+d})\n")
            thrust = ad.get('breadth_thrust', 0.5)
            if thrust > 0.70 or thrust < 0.30:
                lines.append(f"- **Breadth Thrust extremo:** {thrust*100:.1f}%\n")
            lines.append(f"- **A/D Line (acumulada):** {ad.get('ad_line', 0):.0f}\n")

        mte_scenario_conf = confirmation_data.get('mte_scenario', '')
        if mte_scenario_conf == 'RECESSION':
            nh_nl = ad.get('nh_nl', 0)
            if nh_nl < 0:
                lines.append(f"- **RECESSION CAPITULATION SIGNAL:** NH/NL negativo ({nh_nl:+d}). Evidencia preliminar de posible rebote tactico.\n")

        ratios = confirmation_data.get('ratios', {})
        if ratios:
            lines.append("\n### Cross-Asset Ratios\n")
            lines.append("| Ratio | Valor | Delta 20d | Z-Score (60d) |\n")
            lines.append("|-------|-------|-----------|---------------|\n")
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
                    lines.append(f"| {label} | {val:.4f} | {delta_str} | {z_str} |\n")
        lines.append("\n")

    # =========================================================================
    # DARK POOLS
    # =========================================================================
    if darkpool_data:
        lines.append("## Actividad en ATS - Dark Pools (FINRA v1.0)\n")
        lines.append("*Nota: FINRA publica datos de ATS con retraso regulatorio de 2 a 4 semanas. Los datos pueden estar desfasados por diseño.*\n")
        week = darkpool_data.get('week', 'N/A')
        if week != 'N/A':
            try:
                d = pd.Timestamp(week)
                age = (datetime.now() - d).days
                freshness = _classify_finra_freshness(age)
                if freshness == 'ARCHIVAL':
                    lines.append(f"**DATOS OBSOLETOS:** Ultimo dato con {age} dias de antiguedad. No se usa para clasificacion actual. Contexto historico solamente.\n\n")
            except:
                pass
        lines.append(f"- **% Volumen en ATS medio:** {darkpool_data.get('media_dark_pool', 0):.2f}% "
                     f"({darkpool_data.get('n_tickers_ats', 0)}/{darkpool_data.get('n_tickers_total', 0)} tickers)\n")
        
        z_windows = darkpool_data.get('z_windows', {})
        if z_windows:
            lines.append("- **Z-Scores por ventana:**\n")
            for w_name, w_data in z_windows.items():
                if w_data:
                    lines.append(f"  - {w_name}: Z={w_data['z']:.2f}, Estado={w_data['state']}\n")
        elif pd.notna(darkpool_data.get('z_score')):
            lines.append(f"- **Robust Z-Score:** {darkpool_data['z_score']:.2f}\n")
            lines.append(f"- **Momentum:** {darkpool_data.get('momentum', 0):.2f}\n")
            lines.append(f"- **Percentil:** {darkpool_data.get('percentile', 0):.0f}%\n")
            lines.append(f"- **Estado ATS:** {darkpool_data.get('state', 'N/A')}\n")
        else:
            lines.append("- *Acumulando historial (se necesitan {DARKPOOL_FULL_HISTORY_WEEKS} semanas para el Z-Score)*\n")
        if week != 'N/A':
            try:
                d = pd.Timestamp(week)
                age = (datetime.now() - d).days
                lines.append(f"- **Semana FINRA:** {week} (retraso: {age} dias)\n")
            except Exception:
                lines.append(f"- **Semana FINRA:** {week}\n")
        else:
            lines.append("- **Semana FINRA:** N/D\n")

        if 'datos' in darkpool_data and not darkpool_data['datos'].empty:
            lines.append("\n**Mayor % de volumen en ATS:**\n")
            lines.append("| Ticker | % ATS | Vol ATS | Vol Total |\n")
            lines.append("|--------|:-----:|:-------:|:---------:|\n")
            top5 = darkpool_data['datos'].nlargest(5, 'dark_pool_pct')
            for _, row in top5.iterrows():
                lines.append(f"| {row['ticker']} | {row['dark_pool_pct']:.2f}% | {row['ats_volume']:,.0f} | {row['total_volume']:,.0f} |\n")
            lines.append("\n*Nota: Un alto % de volumen en ATS NO implica acumulación institucional. Las categorias reflejan el nivel de actividad ATS relativa a su historial, no la direccion del flujo institucional.*\n")
        lines.append("\n*Fuente: FINRA ATS Transparency Data.*\n\n")

    # =========================================================================
    # INFERENCIA TRANSVERSAL (CORREGIDA)
    # =========================================================================
    lines.append("")
    
    # =====================================================================
    # ESTADO ACTUAL — SÍNTESIS DE SEÑALES (v3.15 corregido)
    # Solo presenta estados oficiales de los módulos. No infiere causas.
    # Máximo 3 elementos, sin especulación, sin redundancias.
    # =====================================================================
    # INDICES INTERNACIONALES — OPORTUNIDADES DE ACUMULACION
    # =====================================================================
    # -----------------------------------------------------------------
    # INDICES INTERNACIONALES — FASES WYCKOFF
    # -----------------------------------------------------------------
    lines.append("\n## Indices Internacionales — Fases Wyckoff\n")
    lines.append("| Indice | Ticker | Fase Wyckoff |\n")
    lines.append("|--------|--------|--------------|\n")
    if index_phases:
        for nombre, fase in index_phases.items():
            ticker = INDEX_CONFIG.get(nombre, {}).get('index_ticker', '')
            lines.append(f"| {nombre} | {ticker} | {fase} |\n")
    else:
        lines.append("| No disponible | No disponible | No disponible |\n")
    lines.append("\n")

    lines.append("\n## Indices Internacionales — Oportunidades de Acumulación y Markup\n")
    lines.append("*Nota: Los componentes se obtienen de ETFs proxy que replican el indice de referencia. Solo se muestran indices en fase ACCUMULATION o MARKUP.*\n\n")
    if index_leaders:
        for nombre, top5 in index_leaders.items():
            if top5 is None or top5.empty:
                continue
            lines.append(f"### {nombre}\n")
            lines.append("*Nota: Flujo (z) es un z-score robusto sobre 60 días. Valores extremos pueden deberse a eventos corporativos o volúmenes inusuales. Para el WLS se usa una versión normalizada limitada a ±3.*\n\n")
            lines.append("| # | Ticker | RS | RS Mom | Flujo (z) | WLS | Fase Wyckoff |\n")
            lines.append("|---|--------|----|--------|-----------|-----|---------------|\n")
            for i, (_, row) in enumerate(top5.iterrows(), 1):
                lines.append(f"| {i} | {row['ticker']} | {row['rs']:.2f} | {row['rs_mom']:.2%} | {row['flow_proxy_z']:.2f} | {row['wls']:.2f} | {row['wyckoff_phase']} |\n")
            lines.append("\n")
    else:
        lines.append("*Ningún indice en fase de acumulación o markup en esta ejecución.*\n\n")

    lines.append("## Estado Actual — Síntesis de Señales\n\n")

    resumen = []

    # 1. Régimen macro (prioridad máxima)
    if macro_regime in ('RECESSION', 'LIQUIDITY CRISIS', 'STAGFLATION'):
        resumen.append(f"- **Régimen macro: {macro_regime}** — entorno de estrés elevado.")
    elif macro_regime in ('EXPANSION', 'RECOVERY', 'GOLDILOCKS'):
        resumen.append(f"- **Régimen macro: {macro_regime}** — favorable para la asunción de riesgo.")
    elif macro_regime == 'MIXED':
        # Leer dispersion real del ultimo dia (fix C20: texto dinamico)
        disp_txt = 'variable'
        try:
            if sector_dispersion_data is not None and not sector_dispersion_data.empty:
                last = sector_dispersion_data.iloc[-1]
                lectura = last.get('dispersion_reading') or last.get('Lectura')
                if lectura and isinstance(lectura, str):
                    disp_txt = lectura.lower()
        except Exception as e:
            print(f"  [WARN] report_generator: sector_dispersion_data: {e}")
        resumen.append(f"- **Régimen macro: MIXED** — ROTATIONAL / MIXED — rotación sectorial activa con dispersión {disp_txt}.")
    else:
        resumen.append(f"- **Régimen macro: {macro_regime}**.")

    # 2. Liderazgo sectorial (prioridad alta)
    if slpm_v12_data:
        leader = slpm_v12_data.get('sector', '')
        state = slpm_v12_data.get('state', '')
        if leader and state:
            if state == 'CONFIRMED':
                resumen.append(f"- **Liderazgo confirmado: {leader}** (SLPM: CONFIRMED).")
            elif state == 'UNRESOLVED':
                resumen.append(f"- **Liderazgo no confirmado: {leader}** (#1 del ranking, SLPM: UNRESOLVED).")
            else:
                resumen.append(f"- **Liderazgo sectorial: {leader}** (SLPM: {state}).")

    # 3. Condiciones financieras o liquidez (si es relevante)
    if liquidity_regime in ('HIGH_STRESS', 'EXTREME_STRESS'):
        resumen.append(f"- **Condiciones financieras: {liquidity_regime}** — estrés elevado en crédito y liquidez.")
    elif liquidity_regime == 'ESTRECHA':
        resumen.append("- **Condiciones financieras: ESTRECHA** — señales financieras en territorio restrictivo.")

    # Máximo 3 elementos
    for item in resumen[:3]:
        lines.append(item + "\n")
    lines.append("\n")

    # Divergencias relevantes (solo si no están ya en el resumen)
    divergencias = []
    if breadth_values:
        ema200 = breadth_values.get('% sobre EMA200', 0)
        ema20 = breadth_values.get('% sobre EMA20', 0)
        if ema200 > 0.70 and ema20 < 0.60:
            divergencias.append(f"- **Breadth Divergence:** Breadth EMA200: {ema200:.0%}; Breadth EMA20: {ema20:.0%}. La amplitud de corto plazo es inferior a la de largo plazo.")
    if price_flow_divergences:
        for ticker, div in price_flow_divergences.items():
            if div.get('status') == 'PRICE_STRONG_FLOW_UNCONFIRMED':
                name = SECTOR_NAMES.get(ticker, ticker)
                divergencias.append(f"- **{name}**: precio fuerte sin confirmación del Flow Proxy.")
    if divergencias and len(resumen) < 3:
        lines.append("### Divergencias Relevantes (Síntesis)\n")
        for d in divergencias[:2]:
            lines.append(d + "\n")
        lines.append("\n")
    # Nota de cierre
    lines.append("*Esta sección describe únicamente estados observables del sistema. No interpreta causas ni sugiere acciones.*\n\n")
    
    lines.append("\n*Esta interpretacion es descriptiva y no constituye una recomendacion de inversion.*\n\n")

    if evidence_matrix_data is not None and not evidence_matrix_data.empty:
        def _fmt_evidence(v):
            if pd.isna(v):
                return 'NA'
            return f"{int(v):+d}"

        lines.append("\n## Matriz de Evidencia\n\n")
        lines.append("| Sector | Precio | Amplitud | Flujo 1º | Flujo Proxy | Wyckoff | Crédito* | Volat* | Calidad | Lectura |\n")
        lines.append("|--------|--------|----------|----------|-------------|---------|----------|--------|---------|----------|\n")
        for _, row in evidence_matrix_data.iterrows():
            lines.append(
                f"| {row['sector']} | {_fmt_evidence(row['price_evidence'])} | {_fmt_evidence(row['breadth_evidence'])} | {_fmt_evidence(row['primary_flow_evidence'])} | {_fmt_evidence(row['proxy_flow_evidence'])} | {_fmt_evidence(row['wyckoff_evidence'])} | {_fmt_evidence(row['credit_evidence'])} | {_fmt_evidence(row['volatility_evidence'])} | {row['evidence_quality']} | {row['alignment_reading']} |\n"
            )
        lines.append("\n*Crédito y volatilidad representan contexto común de mercado y no participan en el balance de evidencia sectorial.*\n")
        lines.append("\n*La ausencia de Flow Proxy (NaN) representa ausencia de evidencia disponible y no se interpreta como neutralidad.*\n")
        lines.append("\n")

    if dc_summary:
        lines.append(dc_summary)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    hist_path = 'outputs/history/macro_regime.csv'
    new_row = pd.DataFrame({
        'date': [datetime.now()],
        'macro_regime': [macro_regime],
        'macro_score': [macro_score.iloc[-1]],
        'macro_conf': [macro_conf],
        'liquidity_regime': [liquidity_regime],
        'volatility_regime': [vol_regime],
        'sector_regime': [sector_regime],
    })
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
        hist = pd.concat([hist, new_row], ignore_index=True)
    else:
        hist = new_row
    hist.to_csv(hist_path, index=False)

    sector_df = pd.DataFrame(sector_results['ranking'], columns=['ticker', 'name', 'score', 'wyckoff_phase'])
    sector_df.to_csv('outputs/report/sector_rankings.csv', index=False)

