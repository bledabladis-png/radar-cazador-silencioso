import os
from datetime import datetime
from zoneinfo import ZoneInfo
from src.report.iae import render_iae_section
from src.report.alerts import render_alerts, render_cross_module
from src.report.breadth import render_breadth_market
from src.report.freshness import render_data_freshness, render_calidad_datos
from src.report.sectorial import (
    render_sector_breadth,
    render_sector_concentration,
    render_sector_dispersion,
)
from src.report.leaders import (
    render_momentum_sectores,
    render_momentum_otros,
)
from src.report.rankings import (
    render_ranking_sectorial_unificado,
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
from src.report.wyckoff_holdings import (
    render_wyckoff_holdings_acc,
    render_wyckoff_holdings_dist,
)
from src.report.sector_context import (
    render_matriz_regimen,
    render_wyckoff_sectorial,
    render_momentum_amplitud,
)
from src.report.flows_international import (
    render_flujos_internacionales_agregados,
    render_flujo_qqq_sec,
    render_posicionamiento_cftc,
    render_flujo_posicional_nport,
    render_rendimiento_qqq,
    render_qqq_nport_flow,
    render_flujo_sintesis,
)
from src.report.sentiment import render_sentimiento_opciones
from src.report.volatility_mte import (
    render_estructura_volatilidad,
    render_mte,
)
from src.report.confirmation import render_confirmation
from src.report.darkpool import render_darkpool
from src.report.synthesis import (
    render_sintesis_senales,
    render_matriz_evidencia,
)
from src.report.header import render_regimenes
from src.report.provenance import render_provenance

MODEL_VERSION = "4.3"
WEIGHTS_VERSION = "3"
INDICATORS_VERSION = "2"


def generate_daily_report(macro_score, macro_regime, macro_conf, liquidity_score, financial_regime, liq_conf,
                          volatility_score, vol_regime, vol_conf, sector_results,
                          sector_price_rank, sector_flow_rank, otros_price_rank, otros_flow_rank,
                          breadth_values=None, real_liquidity_regime=None, real_liquidity_conf=None,
                          pcr_data=None, darkpool_data=None, mte_result=None, confirmation_data=None,
                          tactical_scores=None, structural_scores=None,
                          sector_persistence=None, signal_agreements=None, signal_agreements_display=None,
                          cross_module_conflict=None, shock_sensitivities=None, price_flow_divergences=None,
                          dc_summary="", all_signals=None, real_liq_score=None, real_liq_prev=None, etf_primary_flow_data=None, cftc_position_flow_data=None, flow_synthesis=None, blackrock_dax_flow=None, blackrock_isf_flow=None, amundi_lyxi_flow=None, blackrock_iwm_flow=None, nport_position_change_data=None, qqq_performance_data=None, qqq_nport_flow_data=None, qqq_sec_flow=None, sector_breadth_data=None, sector_breadth_is_stale=False, sector_breadth_stale_reason=None, sector_concentration_data=None, sector_flow_characteristics_data=None, rs_internal_data=None, sector_rank_deltas_data=None, sector_regime_matrix_data=None, sector_wyckoff_distribution_data=None, sector_breadth_momentum_data=None, evidence_matrix_data=None, sector_dispersion_data=None, sector_correlation_summary_data=None, cross_asset_context_data=None, volatility_structure_data=None, data_quality_data=None, iae_section=None, output_path='outputs/report/reporte_diario.md',
                          reference_date=None, provenance=None,
                          wyckoff_holdings_data=None):
    lines = []
    lines.append("# MACRO SECTORIAL - Reporte Diario\n")
    # D6 (2026-09-30): la fecha del header se deriva del dataset, no
    # de datetime.now(). run.py resuelve reference_date UNA vez al
    # inicio (tz-aware, Europe/Madrid) y la propaga. Fallback a now()
    # con tz explicita si no llega (uso aislado de la funcion).
    _ref = reference_date if reference_date is not None else datetime.now(ZoneInfo('Europe/Madrid'))
    # H1 (2026-10-17, REP-AUDIT-20261017-capa3): separar los dos
    # timestamps que el reporte mezclaba bajo '**Fecha:**'. El
    # auditor (2026-10-17) detecto que reference_date es el inicio
    # del run, no el momento de generacion. Ahora se etiquetan
    # explicitamente:
    #   - Referencia: instante de corte del pipeline (reference_date).
    #   - Generado: momento real de generacion (provenance[generated_at]).
    lines.append(f"**Referencia:** {_ref.strftime('%Y-%m-%d %H:%M:%S')}\n")
    if provenance and isinstance(provenance, dict) and provenance.get("generated_at"):
        lines.append(f"**Generado:** {provenance['generated_at']} (UTC)\n")
    lines.append(f"**Modelo:** v{MODEL_VERSION} | Pesos: v{WEIGHTS_VERSION} | Indicadores: v{INDICATORS_VERSION}\n\n")

    # H8 (2026-10-16, auditoria externa): bloque de provenance.
    # Identifica commit, run de CI, sesion objetivo y fechas efectivas.
    # Permite certificar que la salida corresponde a un commit concreto.
    lines.extend(render_provenance(provenance))

    # =========================================================================
    # RESUMEN DE REGIMENES (extraido a src/report/header.py, C1-5)
    # =========================================================================
    lines.extend(render_regimenes(
        macro_score, macro_regime, macro_conf,
        liquidity_score, financial_regime, liq_conf,
        volatility_score, vol_regime, vol_conf,
        real_liquidity_regime, real_liquidity_conf,
        real_liq_score, real_liq_prev,
        sector_regime=sector_results['regime'],
        confirmation_data=confirmation_data
    ))
    # =========================================================================
    # DATA FRESHNESS + ALERTAS + CROSS-MODULE
    # =========================================================================
    lines.extend(render_data_freshness(pcr_data, darkpool_data, sector_results, reference_date=reference_date))
    lines.extend(render_alerts(breadth_values, financial_regime, price_flow_divergences, confirmation_data))
    lines.extend(render_cross_module(cross_module_conflict))

    # =========================================================================
    # C1 - REGIMEN GLOBAL
    # =========================================================================
    lines.extend(render_breadth_market(breadth_values))
    lines.extend(render_estructura_volatilidad(volatility_structure_data))
    lines.extend(render_calidad_datos(data_quality_data))
    lines.extend(render_mte(mte_result))
    lines.extend(render_confirmation(confirmation_data))
    lines.extend(render_darkpool(darkpool_data, reference_date=reference_date))
    lines.extend(render_sentimiento_opciones(pcr_data, reference_date=reference_date))

    # =========================================================================
    # C2 - ANALISIS SECTORIAL
    # =========================================================================
    lines.extend(render_sector_breadth(sector_breadth_data,
                                       is_stale=sector_breadth_is_stale,
                                       stale_reason=sector_breadth_stale_reason))
    lines.extend(render_sector_concentration(sector_concentration_data))
    lines.extend(render_sector_dispersion(sector_concentration_data))
    lines.extend(render_ranking_sectorial_unificado(
        sector_results, tactical_scores, structural_scores,
        sector_persistence, signal_agreements, signal_agreements_display,
        shock_sensitivities,
    ))
    lines.extend(render_opportunity_map(
        tactical_scores, structural_scores,
        sector_persistence, signal_agreements,
    ))
    # C2 - contexto sectorial interno
    lines.extend(render_matriz_regimen(sector_regime_matrix_data))
    lines.extend(render_wyckoff_sectorial(sector_wyckoff_distribution_data))
    lines.extend(render_wyckoff_holdings_acc(wyckoff_holdings_data))
    lines.extend(render_wyckoff_holdings_dist(wyckoff_holdings_data))
    lines.extend(render_momentum_amplitud(sector_breadth_momentum_data))
    lines.extend(render_liderazgo_interno(rs_internal_data))
    lines.extend(render_rotacion_reciente(sector_rank_deltas_data))
    lines.extend(render_dispersion_sectores(sector_dispersion_data))
    lines.extend(render_correlacion_sectores(sector_correlation_summary_data))
    lines.extend(render_contexto_cross_asset(cross_asset_context_data))

    # C2 - momentum y flow proxy (sectores + otros activos)
    lines.extend(render_momentum_sectores(sector_price_rank, sector_flow_rank))
    lines.extend(render_momentum_otros(otros_price_rank, otros_flow_rank))

    # =========================================================================
    # C4 - CONFIRMACION
    # =========================================================================
    lines.extend(render_matriz_evidencia(evidence_matrix_data))

    # =========================================================================
    # C5 - CONTEXTO INTERNACIONAL
    # =========================================================================
    # Flujos primarios ETF (SPDR diario)
    lines.extend(render_flujo_spdr(etf_primary_flow_data))
    lines.extend(render_flujo_caracteristicas(sector_flow_characteristics_data))
    lines.extend(render_divergencia_precio_flujo(sector_flow_characteristics_data))
    # Flujos primarios internacionales (diarios)
    lines.extend(render_flujos_internacionales_agregados(
        blackrock_dax_flow, blackrock_isf_flow,
        amundi_lyxi_flow, blackrock_iwm_flow,
    ))
    # Flujos QQQ SEC / CFTC / N-PORT / Rendimiento QQQ / QQQ NPORT
    lines.extend(render_flujo_qqq_sec(qqq_sec_flow))
    lines.extend(render_posicionamiento_cftc(cftc_position_flow_data))
    lines.extend(render_flujo_posicional_nport(nport_position_change_data))
    lines.extend(render_rendimiento_qqq(qqq_performance_data))
    lines.extend(render_qqq_nport_flow(qqq_nport_flow_data))
    lines.extend(render_flujo_sintesis(flow_synthesis))
    # Estado actual y 13F
    lines.extend(render_sintesis_senales(
        macro_regime, financial_regime,
        sector_dispersion_data, breadth_values, price_flow_divergences,
    ))
    if dc_summary:
        lines.append(dc_summary)
    lines.extend(render_iae_section(iae_section))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

# NOTE: side effects (macro_regime.csv, sector_rankings.csv) movidos a run.py (C1-10)

