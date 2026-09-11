# -*- coding: utf-8 -*-
"""
Macro Sectorial v4.3 -- Sistema de analisis macro y rotacion sectorial.
Fases 1-4 + Correccion 0.5 + P1 + P2 + Mejoras 16-20.
"""
import os
import sys
from src.report_generator import generate_daily_report
from src.pipeline.data_load import load_all_data
from src.pipeline.regimes import compute_all_regimes
from src.pipeline.sectors_base import compute_sectors_base
from src.pipeline.flows_primary import compute_flows_primary
from src.pipeline.flows_secondary import compute_flows_secondary
from src.pipeline.leaders import compute_leaders
from src.pipeline.sector_metrics import compute_sector_metrics
from src.pipeline.breadth_metrics import compute_breadth_metrics
from src.pipeline.engines import compute_engines
from src.pipeline.slpm import compute_slpm_v12
from src.pipeline.diagnostics import compute_diagnostics
from src.pipeline.market_data import compute_market_data
from src.pipeline.mte_confirmation import compute_mte_confirmation
from src.pipeline.indices_intl import compute_indices_intl
from src.pipeline.validation_gate import run_validation_gate
from src.pipeline.finalize import (
    compute_final_matrices,
    save_regime_history,
    save_sector_rankings,
    generate_european_coverage,
)
from config.tickers import validate_sector_universe

def main():
    validate_sector_universe()
    # Crear subcarpetas de outputs necesarias para ejecución limpia
    for subdir in ['report', 'history', 'state', 'holdings', 'audit', 'cache']:
        os.makedirs(f'outputs/{subdir}', exist_ok=True)
    # Fix C22: limpiar CSVs de lideres previos. Si el run actual no los
    # regenera (sin sectores favorables / sin indices en fase), los
    # consumidores (report_generator, run_all_audits, verify_leader_selection)
    # no deben leer datos obsoletos de runs anteriores.
    for _csv_name in ['analisis_lideres.csv', 'analisis_lideres_internacionales.csv']:
        _csv_path = os.path.join('outputs', 'report', _csv_name)
        if os.path.exists(_csv_path):
            try:
                os.remove(_csv_path)
            except Exception:
                pass
    data = load_all_data()
    if data is None:
        return
    df_market = data['df_market']
    df_macro_manual = data['df_macro_manual']

    regimes = compute_all_regimes(df_market, df_macro_manual)
    financial_score = regimes['financial_score']
    financial_regime = regimes['financial_regime']
    liq_conf = regimes['liq_conf']
    real_liq_score = regimes['real_liq_score']
    real_liq_regime = regimes['real_liq_regime']
    real_liq_conf = regimes['real_liq_conf']
    real_liq_prev = regimes['real_liq_prev']
    vol_score = regimes['vol_score']
    vol_regime = regimes['vol_regime']
    vol_conf = regimes['vol_conf']
    macro_score = regimes['macro_score']
    macro_regime = regimes['macro_regime']
    macro_conf = regimes['macro_conf']
    all_signals = regimes['all_signals']

    sb = compute_sectors_base(df_market)
    sector_results = sb['sector_results']
    sector_rank_deltas_df = sb['sector_rank_deltas_df']
    sector_price_rank = sb['sector_price_rank']
    sector_flow_rank = sb['sector_flow_rank']
    otros_price_rank = sb['otros_price_rank']
    otros_flow_rank = sb['otros_flow_rank']
    sector_dispersion_df = sb['sector_dispersion_df']
    sector_corr_summary_df = sb['sector_corr_summary_df']
    sector_corr_matrix_df = sb['sector_corr_matrix_df']
    cross_asset_summary_df = sb['cross_asset_summary_df']
    breadth_values = sb['breadth_values']

    fp = compute_flows_primary(df_market)
    etf_primary_flow_data = fp['etf_primary_flow_data']
    sector_flow_characteristics_df = fp['sector_flow_characteristics_df']
    blackrock_dax_flow = fp['blackrock_dax_flow']
    blackrock_isf_flow = fp['blackrock_isf_flow']
    blackrock_iwm_flow = fp['blackrock_iwm_flow']
    amundi_lyxi_flow = fp['amundi_lyxi_flow']
    qqq_sec_flow = fp['qqq_sec_flow']
    cftc_position_flow_data = fp['cftc_position_flow_data']

    fs = compute_flows_secondary(
        sector_flow_rank, etf_primary_flow_data, cftc_position_flow_data,
        blackrock_dax_flow, blackrock_isf_flow, amundi_lyxi_flow,
    )
    flow_synthesis = fs['flow_synthesis']
    nport_position_change_data = fs['nport_position_change_data']
    qqq_performance_data = fs['qqq_performance_data']
    qqq_nport_flow_data = fs['qqq_nport_flow_data']

    ldr = compute_leaders(df_market, sector_results)
    df_stocks = ldr['df_stocks']
    holdings_df = ldr['holdings_df']
    leader_lines = ldr['leader_lines']
    leader_df = ldr['leader_df']
    full_metrics_df = ldr['full_metrics_df']

    sm = compute_sector_metrics(df_stocks, holdings_df, leader_df, full_metrics_df, df_market)
    sector_leader_divergence_df = sm['sector_leader_divergence_df']
    sector_wyckoff_distribution_df = sm['sector_wyckoff_distribution_df']
    rs_internal_df = sm['rs_internal_df']
    sector_concentration_df = sm['sector_concentration_df']
    leader_representativeness_df = sm['leader_representativeness_df']

    bm = compute_breadth_metrics(df_stocks, df_market, holdings_df)
    sector_breadth_momentum_df = bm['sector_breadth_momentum_df']
    sector_breadth_df = bm['sector_breadth_df']

    en = compute_engines(df_market, sector_results, sector_flow_rank, otros_flow_rank, leader_df)
    leader_metrics_for_slpm = en['leader_metrics_for_slpm']
    top_sector_flow = en['top_sector_flow']
    tactical_scores = en['tactical_scores']
    structural_scores = en['structural_scores']
    sector_persistence = en['sector_persistence']

    slpm_v12_data = compute_slpm_v12(
        df_market, sector_results, leader_metrics_for_slpm,
        top_sector_flow, tactical_scores, structural_scores,
        sector_persistence,
    )

    diag = compute_diagnostics(df_market, tactical_scores, structural_scores, sector_flow_rank)
    signal_agreements = diag['signal_agreements']
    signal_agreements_display = diag['signal_agreements_display']
    price_flow_divergences = diag['price_flow_divergences']
    shock_sensitivities = diag['shock_sensitivities']

    md = compute_market_data(df_market)
    pcr_data = md['pcr_data']
    darkpool_data = md['darkpool_data']
    vol_structure_df = md['vol_structure_df']
    data_quality_df = md['data_quality_df']

    mc = compute_mte_confirmation(
        df_market, df_stocks, financial_score, all_signals,
        pcr_data, darkpool_data, macro_regime,
        financial_regime, vol_regime, real_liq_regime,
    )
    mte_result = mc['mte_result']
    cross_module_conflict = mc['cross_module_conflict']
    confirmation_data = mc['confirmation_data']

    ii = compute_indices_intl(df_market)
    index_phases = ii['index_phases']
    index_leaders = ii['index_leaders']

    vg = run_validation_gate(
        slpm_v12_data, pcr_data, darkpool_data, mte_result,
        tactical_scores, structural_scores,
    )
    if not vg['passed']:
        sys.exit(1)
    dc_summary = vg['dc_summary']

    mats = compute_final_matrices(
        sector_breadth_df, sector_concentration_df,
        sector_flow_characteristics_df, sector_wyckoff_distribution_df,
        sector_results, financial_regime, real_liq_regime,
        vol_regime, vol_score, real_liq_score, financial_score,
    )
    sector_regime_matrix_df = mats['sector_regime_matrix_df']
    evidence_matrix_df = mats['evidence_matrix_df']

    print("Generando reporte...")
    generate_daily_report(macro_score, macro_regime, macro_conf,
                          financial_score, financial_regime, liq_conf,
                          vol_score, vol_regime, vol_conf,
                          sector_results,
                          sector_price_rank, sector_flow_rank, otros_price_rank, otros_flow_rank,
                          leader_lines=leader_lines, breadth_values=breadth_values,
                            etf_primary_flow_data=etf_primary_flow_data,
                            blackrock_dax_flow=blackrock_dax_flow,
                            blackrock_isf_flow=blackrock_isf_flow,
                            blackrock_iwm_flow=blackrock_iwm_flow,
                            amundi_lyxi_flow=amundi_lyxi_flow,
                            nport_position_change_data=nport_position_change_data,
                            qqq_performance_data=qqq_performance_data,
                            qqq_nport_flow_data=qqq_nport_flow_data,
                            cftc_position_flow_data=cftc_position_flow_data,
                            qqq_sec_flow=qqq_sec_flow,
                            flow_synthesis=flow_synthesis,
                          real_liquidity_regime=real_liq_regime, real_liquidity_conf=real_liq_conf,
                            real_liq_score=real_liq_score,
                          pcr_data=pcr_data, darkpool_data=darkpool_data, mte_result=mte_result,
                          confirmation_data=confirmation_data,
                          slpm_v12_data=slpm_v12_data,
                          tactical_scores=tactical_scores,
                          structural_scores=structural_scores,
                          sector_persistence=sector_persistence,
                          signal_agreements=signal_agreements,
                          signal_agreements_display=signal_agreements_display,
                          cross_module_conflict=cross_module_conflict,
                          shock_sensitivities=shock_sensitivities,
                          price_flow_divergences=price_flow_divergences,
                          dc_summary=dc_summary,
                          real_liq_prev=real_liq_prev, index_leaders=index_leaders, index_phases=index_phases, sector_breadth_data=sector_breadth_df, sector_concentration_data=sector_concentration_df, sector_flow_characteristics_data=sector_flow_characteristics_df, rs_internal_data=rs_internal_df, sector_rank_deltas_data=sector_rank_deltas_df, sector_regime_matrix_data=sector_regime_matrix_df, leader_representativeness_data=leader_representativeness_df, sector_wyckoff_distribution_data=sector_wyckoff_distribution_df, sector_leader_divergence_data=sector_leader_divergence_df, sector_breadth_momentum_data=sector_breadth_momentum_df,
                          evidence_matrix_data=evidence_matrix_df,
                          sector_dispersion_data=sector_dispersion_df,
                          sector_correlation_summary_data=sector_corr_summary_df,
                          sector_correlation_matrix_data=sector_corr_matrix_df,
                          cross_asset_context_data=cross_asset_summary_df,
                          volatility_structure_data=vol_structure_df,
                          data_quality_data=data_quality_df,

                          all_signals=all_signals)
    print("Reporte generado en outputs/report/reporte_diario.md")

    # Side effects + cobertura europea (C1-10 + C2-12)
    save_regime_history(macro_score, macro_regime, macro_conf,
                        financial_regime, vol_regime, sector_results)
    save_sector_rankings(sector_results)
    generate_european_coverage()


if __name__ == "__main__":
    main()

















