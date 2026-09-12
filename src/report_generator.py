import os
from datetime import datetime
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
from src.report.darkpool import render_darkpool
from src.report.synthesis import (
    render_indices_internacionales,
    render_sintesis_senales,
    render_matriz_evidencia,
)
from src.report.header import render_regimenes

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
                          dc_summary="", all_signals=None, real_liq_score=None, real_liq_prev=None, index_leaders=None, index_phases=None, etf_primary_flow_data=None, cftc_position_flow_data=None, flow_synthesis=None, blackrock_dax_flow=None, blackrock_isf_flow=None, amundi_lyxi_flow=None, blackrock_iwm_flow=None, nport_position_change_data=None, qqq_performance_data=None, qqq_nport_flow_data=None, qqq_sec_flow=None, sector_breadth_data=None, sector_breadth_is_stale=False, sector_concentration_data=None, sector_flow_characteristics_data=None, rs_internal_data=None, sector_rank_deltas_data=None, sector_regime_matrix_data=None, leader_representativeness_data=None, sector_wyckoff_distribution_data=None, sector_leader_divergence_data=None, sector_breadth_momentum_data=None, evidence_matrix_data=None, sector_dispersion_data=None, sector_correlation_summary_data=None, sector_correlation_matrix_data=None, cross_asset_context_data=None, volatility_structure_data=None, data_quality_data=None, output_path='outputs/report/reporte_diario.md'):
    lines = []
    lines.append("# MACRO SECTORIAL - Reporte Diario\n")
    lines.append(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Modelo:** v{MODEL_VERSION} | Pesos: v{WEIGHTS_VERSION} | Indicadores: v{INDICATORS_VERSION}\n\n")

    # =========================================================================
    # RESUMEN DE REGIMENES (extraido a src/report/header.py, C1-5)
    # =========================================================================
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
    lines.extend(render_sector_breadth(sector_breadth_data, is_stale=sector_breadth_is_stale))

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
    # CONFIRMATION DATA (Nivel 2)
    # =========================================================================
    lines.extend(render_confirmation(confirmation_data))

    # =========================================================================
    # DARK POOLS
    # =========================================================================
    # DARK POOLS
    # =========================================================================
    lines.extend(render_darkpool(darkpool_data))

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
    # =========================================================================
    # INDICES INTERNACIONALES
    # =========================================================================
    lines.extend(render_indices_internacionales(index_phases, index_leaders))

    # =========================================================================
    # SINTESIS DE SEÑALES
    # =========================================================================
    lines.extend(render_sintesis_senales(
        macro_regime, slpm_v12_data, liquidity_regime,
        sector_dispersion_data, breadth_values, price_flow_divergences,
    ))

    # =========================================================================
    # MATRIZ DE EVIDENCIA
    # =========================================================================
    lines.extend(render_matriz_evidencia(evidence_matrix_data))

    if dc_summary:
        lines.append(dc_summary)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

# NOTE: side effects (macro_regime.csv, sector_rankings.csv) movidos a run.py (C1-10)

