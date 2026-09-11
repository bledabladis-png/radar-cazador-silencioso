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
from src.report.flows_international import (
    render_flujo_daxex,
    render_flujo_isf,
    render_flujo_lyxi,
    render_flujo_iwm,
    render_flujo_qqq_sec,
    render_posicionamiento_cftc,
    render_flujo_posicional_nport,
    render_rendimiento_qqq,
    render_qqq_nport_flow,
    render_flujo_sintesis,
)
from src.report.sentiment import render_sentimiento_opciones
from src.report.slpm import render_slpm_v12, render_slpm_legacy
from src.report.volatility_mte import (
    render_estructura_volatilidad,
    render_calidad_datos,
    render_mte,
)
from src.report.confirmation import render_confirmation
from src.report.header import render_regimenes
from src.report.helpers import _classify_finra_freshness

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

    # =========================================================================
    # FLUJOS INTERNACIONALES (DAXEX, ISF, LYXI, IWM, QQQ SEC, CFTC)
    # =========================================================================
    lines.extend(render_flujo_daxex(blackrock_dax_flow))
    lines.extend(render_flujo_isf(blackrock_isf_flow))
    lines.extend(render_flujo_lyxi(amundi_lyxi_flow))
    lines.extend(render_flujo_iwm(blackrock_iwm_flow))
    lines.extend(render_flujo_qqq_sec(qqq_sec_flow))
    lines.extend(render_posicionamiento_cftc(cftc_position_flow_data))


    # =========================================================================
    # FLUJO POSICIONAL N-PORT (Trimestral)
    # =========================================================================
    # N-PORT + QQQ Performance + QQQ NPORT-P + Sintesis
    # =========================================================================
    lines.extend(render_flujo_posicional_nport(nport_position_change_data))
    lines.extend(render_rendimiento_qqq(qqq_performance_data))
    lines.extend(render_qqq_nport_flow(qqq_nport_flow_data))
    lines.extend(render_flujo_sintesis(flow_synthesis))

    # =========================================================================
    # =========================================================================
    # VOLATILIDAD + CALIDAD + MTE
    # =========================================================================
    lines.extend(render_estructura_volatilidad(volatility_structure_data))
    lines.extend(render_calidad_datos(data_quality_data))
    lines.extend(render_mte(mte_result))

    # =========================================================================
    # CONFIRMATION DATA
    # =========================================================================
    # CONFIRMATION DATA (Nivel 2)
    # =========================================================================
    lines.extend(render_confirmation(confirmation_data))

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

