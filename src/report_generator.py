import pandas as pd
from src.utils import safe_mean
import numpy as np
import os
from datetime import datetime
from config.tickers import SECTOR_NAMES
from config.index_tickers import INDEX_CONFIG
from config.settings import MOMENTUM_PRICE_WINDOW, MOMENTUM_LONG_WINDOW, ETF_PRIMARY_FLOW_ZSCORE_WINDOW, MIN_SECTOR_COVERAGE
from config.weights import SLPM_WEIGHTS
from src.report.alerts import render_alerts, render_cross_module
from src.report.breadth import render_breadth_market
from src.report.freshness import render_data_freshness
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
    # SECTOR BREADTH & HEALTH
    # =========================================================================
    if sector_breadth_data is not None and not sector_breadth_data.empty:
        latest_date = pd.to_datetime(sector_breadth_data['date']).max()
        breadth_latest = sector_breadth_data[pd.to_datetime(sector_breadth_data['date']) == latest_date]
        lines.append("## Sector Breadth & Health\n")
        lines.append("| Sector | EMA20 | EMA50 | EMA200 | RS+ | Mom+ | Acc | Markup | Dist | Markdown | NH | NL | A/D | Cobertura |\n")
        lines.append("|--------|-------|-------|--------|-----|------|-----|--------|------|----------|----|----|-----|-----------|\n")
        for _, row in breadth_latest.iterrows():
            cobertura = (row['n_valid_ema200'] / row['n_total'] * 100) if row['n_total'] else 0
            cov_label = f"{cobertura:.0f}%"
            if (cobertura / 100) < MIN_SECTOR_COVERAGE:
                cov_label += " [BAJA]"
            lines.append(f"| {row['sector']} | {_fmt_num(row['pct_above_ema20'], '{:.1f}%')} | {_fmt_num(row['pct_above_ema50'], '{:.1f}%')} | {_fmt_num(row['pct_above_ema200'], '{:.1f}%')} | {_fmt_num(row['pct_rs_positive'], '{:.1f}%')} | {_fmt_num(row['pct_momentum_positive'], '{:.1f}%')} | {_fmt_num(row['count_accumulation'], '{:.0f}')} | {_fmt_num(row['count_markup'], '{:.0f}')} | {_fmt_num(row['count_distribution'], '{:.0f}')} | {_fmt_num(row['count_markdown'], '{:.0f}')} | {_fmt_num(row['new_highs'], '{:.0f}')} | {_fmt_num(row['new_lows'], '{:.0f}')} | {_fmt_num(row['ad_net'], '{:+d}')} | {cov_label} |\n")
        lines.append("\n")
        lines.append(f"*[BAJA] = cobertura < {MIN_SECTOR_COVERAGE:.0%} del universo del sector. Los ratios se calculan sobre la parte valida.*\n\n")

    # =========================================================================
    # SECTOR CONCENTRATION
    # =========================================================================
    if sector_concentration_data is not None and not sector_concentration_data.empty:
        latest_date = pd.to_datetime(sector_concentration_data['date']).max()
        conc_latest = sector_concentration_data[pd.to_datetime(sector_concentration_data['date']) == latest_date]
        lines.append("## Concentración del liderazgo\n")
        lines.append("| Sector | Top1 | Top3 | Top5 | RS med | Mom med | Flow med | Wyckoff med | WLS med | Líder | Ret Líder | Cob RS | Cob Mom | Cob Flow | Cob Wyckoff | Cob WLS |\n")
        lines.append("|--------|------|------|------|--------|---------|----------|-------------|---------|-------|----------|--------|---------|----------|-------------|---------|\n")
        for _, row in conc_latest.iterrows():
            lines.append(f"| {row['sector']} | {_fmt_num(row['top1_positive_return_concentration'], '{:.1%}')} | {_fmt_num(row['top3_positive_return_concentration'], '{:.1%}')} | {_fmt_num(row['top5_positive_return_concentration'], '{:.1%}')} | {_fmt_num(row['rs_median'], '{:.4f}')} | {_fmt_num(row['momentum_median'], '{:.2%}')} | {_fmt_num(row['flow_median'], '{:.2f}')} | {_fmt_num(row['wyckoff_median'], '{:.2f}')} | {_fmt_num(row['wls_median'], '{:.2f}')} | {row['leader_ticker']} | {_fmt_num(row['leader_return20'], '{:.2%}')} | {_fmt_num(row['coverage_rs'], '{:.0f}%')} | {_fmt_num(row['coverage_momentum'], '{:.0f}%')} | {_fmt_num(row['coverage_flow'], '{:.0f}%')} | {_fmt_num(row['coverage_wyckoff'], '{:.0f}%')} | {_fmt_num(row['coverage_wls'], '{:.0f}%')} |\n")
        lines.append("\n")
    # =========================================================================
    # DISPERSIÓN INTERNA
    # =========================================================================
    if sector_concentration_data is not None and not sector_concentration_data.empty:
        latest_date = pd.to_datetime(sector_concentration_data['date']).max()
        disp_latest = sector_concentration_data[pd.to_datetime(sector_concentration_data['date']) == latest_date]
        lines.append("## Dispersión interna\n")
        lines.append("| Sector | RS P25 | RS Med | RS P75 | Mom P25 | Mom Med | Mom P75 |\n")
        lines.append("|--------|--------|--------|--------|---------|---------|---------|\n")
        for _, row in disp_latest.iterrows():
            lines.append(f"| {row['sector']} | {row['rs_p25']:.4f} | {row['rs_median']:.4f} | {row['rs_p75']:.4f} | {row['momentum_p25']:.2%} | {row['momentum_median']:.2%} | {row['momentum_p75']:.2%} |\n")
        lines.append("\n")
        lines.append("*Los percentiles de Flow y WLS están disponibles en outputs/history/sector_concentration.csv.*\n\n")

    # TACTICAL LEADERS
    # =========================================================================
    lines.append(f"\n## Momentum de Precio - Sectores ({MOMENTUM_PRICE_WINDOW} dias)\n")
    lines.append("| # | Sector | Retorno 20d (%) |\n")
    lines.append("|---|--------|------------------|\n")
    for i, (ticker, mom) in enumerate(sector_price_rank[:11], 1):
        name = SECTOR_NAMES.get(ticker, ticker)
        lines.append(f"| {i} | {name} ({ticker}) | {mom*100:.2f}% |\n")

    if sector_flow_rank:
        lines.append("\n## Flujo Institucional - Sectores (Proxy)\n")
        lines.append("| # | Sector | Flujo (z-score) |\n")
        lines.append("|---|--------|------------------|\n")
    else:
        lines.append("\n## Flujo Institucional - Sectores (Proxy)\n")
        lines.append("*No hay datos disponibles para Flow Proxy.*\n")
    for i, (ticker, flow) in enumerate(sector_flow_rank[:11], 1):
        name = SECTOR_NAMES.get(ticker, ticker)
        lines.append(f"| {i} | {name} ({ticker}) | {flow:.2f} |\n")

    lines.append("## Tactical Leaders (Momentum de corto plazo)\n")
    lines.append("| # | Sector | Tactical | Structural | Retorno 20d | Flow Proxy (z) | Comm Corr |\n")
    lines.append("|---|--------|----------|------------|-------------|----------------|------------|\n")
    tactical_ranking = sorted(tactical_scores.items(), key=lambda x: x[1], reverse=True) if tactical_scores else []
    for i, (ticker, t_score) in enumerate(tactical_ranking[:11], 1):
        name = SECTOR_NAMES.get(ticker, ticker)
        s_score = structural_scores.get(ticker, 0.0) if structural_scores else 0.0
        mom = next((m for t, m in sector_price_rank if t == ticker), 0)
        flow = next((f for t, f in sector_flow_rank if t == ticker), None)
        shock = shock_sensitivities.get(ticker, {}) if shock_sensitivities else {}
        comm = shock.get('commodity_level', 'N/A') if shock else 'N/A'
        comm_val = shock.get('commodity_corr_value', None) if shock else None
        comm_display = f"{comm} ({comm_val:+.2f})" if comm_val is not None and comm != 'N/A' else comm
        lines.append(f"| {i} | {name} ({ticker}) | {t_score:+.2f} | {s_score:+.2f} | {mom*100:.2f}% | {_fmt_num(flow, '{:+.2f}')} | {comm_display} |\n")
    lines.append("\n")
    lines.append(f"*Nota: Comm Corr mide la correlación de {MOMENTUM_LONG_WINDOW} dias con ^SPGSCI. No implica causalidad.*\n\n")

    # =========================================================================
    # STRUCTURAL RANKING (sin columna Coverage)
    # =========================================================================
    lines.append(f"\n## Momentum de Precio - Otros Activos ({MOMENTUM_PRICE_WINDOW} dias)\n")
    lines.append("| # | Activo | Retorno 20d (%) |\n")
    lines.append("|---|--------|------------------|\n")
    for i, (ticker, mom) in enumerate(otros_price_rank[:15], 1):
        lines.append(f"| {i} | {ticker} | {mom*100:.2f}% |\n")

    if otros_flow_rank:
        lines.append("\n## Flujo Institucional - Otros Activos (Proxy)\n")
        lines.append("| # | Activo | Flujo (z-score) |\n")
        lines.append("|---|--------|------------------|\n")
    else:
        lines.append("\n## Flujo Institucional - Otros Activos (Proxy)\n")
        lines.append("*No hay datos disponibles para Flow Proxy.*\n")
    for i, (ticker, flow) in enumerate(otros_flow_rank[:15], 1):
        lines.append(f"| {i} | {ticker} | {flow:.2f} |\n")

    lines.append("## Structural Ranking (Fortaleza de largo plazo)\n")
    lines.append("| # | Sector | Structural | Tactical | Persist | Agreement | Signal Consistency |\n")
    lines.append("|---|--------|------------|----------|---------|-----------|------------|\n")
    structural_ranking = sorted(structural_scores.items(), key=lambda x: x[1], reverse=True) if structural_scores else []
    for i, (ticker, s_score) in enumerate(structural_ranking[:11], 1):
        name = SECTOR_NAMES.get(ticker, ticker)
        t_score = tactical_scores.get(ticker, 0.0) if tactical_scores else 0.0
        pers_raw = sector_persistence.get(ticker) if sector_persistence else None
        pers_val = pers_raw if pers_raw is not None else 0.0
        pers_str = f"{pers_raw:.0%}" if pers_raw is not None else "N/A"
        agree = signal_agreements.get(ticker, 0.5) if signal_agreements else 0.5
        agree_display = signal_agreements_display.get(ticker, f'{agree:.0%}') if signal_agreements_display else f'{agree:.0%}'
        struct_conf = (pers_val + agree) / 2
        lines.append(f"| {i} | {name} ({ticker}) | {s_score:+.2f} | {t_score:+.2f} | {pers_str} | {agree_display} | {struct_conf:.0%} |\n")
    lines.append("\n")

    # =========================================================================
    # RANKINGS SECTORIALES
    # =========================================================================
    lines.append("## Rankings Sectoriales (Score combinado original)\n")
    lines.append("> *Nota: Este Score es el ranking historico del sistema (momentum, tendencia, volatilidad, breadth, Wyckoff). No es el Tactical ni el Structural Score.*\n\n")
    header = "| # | Sector | Score | Tactical | Structural | Persist | Agreement | Comm Corr | Fase Wyckoff |\n"
    sep = "|---|--------|-------|----------|------------|---------|-----------|------------|---------------|\n"
    lines.append(header)
    lines.append(sep)
    for i, (ticker, name, score, wyckoff) in enumerate(sector_results['ranking'][:11], 1):
        t_score = tactical_scores.get(ticker, 0.0) if tactical_scores else 0.0
        s_score = structural_scores.get(ticker, 0.0) if structural_scores else 0.0
        pers_raw = sector_persistence.get(ticker) if sector_persistence else None
        pers_val = pers_raw if pers_raw is not None else 0.0
        pers_str = f"{pers_raw:.0%}" if pers_raw is not None else "N/A"
        agree = signal_agreements.get(ticker, 0.5) if signal_agreements else 0.5
        agree_display = signal_agreements_display.get(ticker, f'{agree:.0%}') if signal_agreements_display else f'{agree:.0%}'
        shock = shock_sensitivities.get(ticker, {}) if shock_sensitivities else {}
        comm_level = shock.get('commodity_level', 'N/A') if shock else 'N/A'
        comm_val = shock.get('commodity_corr_value', None) if shock else None
        comm_display = f"{comm_level} ({comm_val:+.2f})" if comm_val is not None and comm_level != 'N/A' else comm_level
        lines.append(f"| {i} | {name} ({ticker}) | {score:.2f} | {t_score:+.2f} | {s_score:+.2f} | {pers_str} | {agree_display} | {comm_display} | {wyckoff} |\n")
    lines.append("\n")

    # =========================================================================
    # =========================================================================
    # PERSISTENCIA SECTORIAL
    # =========================================================================
    if sector_persistence:
        lines.append("## Persistencia sectorial\n")
        lines.append("| Sector | Persistencia |\n")
        lines.append("|--------|-------------|\n")
        for ticker in ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']:
            val = sector_persistence.get(ticker)
            val_str = f"{val:.0%}" if val is not None else "N/D"
            lines.append(f"| {ticker} | {val_str} |\n")
        lines.append("\n")
        lines.append("*Persistencia calculada sobre RS20 (acción vs SPY) con lookback 12. Descriptiva, no predictiva.*\n\n")

    # OPPORTUNITY MAP
    # =========================================================================
    lines.append("## Opportunity Map (basado en medianas Tactical/Structural, independiente del SLPM)\n\n")
    tact_values = [v for v in tactical_scores.values() if v is not None] if tactical_scores else [0]
    struct_values = [v for v in structural_scores.values() if v is not None] if structural_scores else [0]
    tact_median = np.median(tact_values) if tact_values else 0
    struct_median = np.median(struct_values) if struct_values else 0
    
    lines.append(f"*Umbrales del dia: Tactical mediana={tact_median:+.2f}, Structural mediana={struct_median:+.2f}*\n\n")
    lines.append("| Cuadrante | Sectores | Signal Consistency |\n")
    lines.append("|-----------|----------|------------|\n")
    
    quadrants = {
        'Structural Strength': [],
        'Tactical Correction': [],
        'Tactical Strength': [],
        'Structural Weakness': [],
        'Transition': []
    }
    
    for ticker in ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']:
        t = tactical_scores.get(ticker, 0) if tactical_scores else 0
        s = structural_scores.get(ticker, 0) if structural_scores else 0
        name = SECTOR_NAMES.get(ticker, ticker)
        pers_raw = sector_persistence.get(ticker) if sector_persistence else None
        pers_val = pers_raw if pers_raw is not None else 0.0
        pers_str = f"{pers_raw:.0%}" if pers_raw is not None else "N/A"
        agree = signal_agreements.get(ticker, 0.5) if signal_agreements else 0.5
        conf = (pers_val + agree) / 2
        
        if s > struct_median and t > tact_median:
            quadrants['Structural Strength'].append((name, conf))
        elif s > struct_median and t < tact_median:
            quadrants['Tactical Correction'].append((name, conf))
        elif s < struct_median and t > tact_median:
            quadrants['Tactical Strength'].append((name, conf))
        elif s < struct_median and t < tact_median:
            quadrants['Structural Weakness'].append((name, conf))
        else:
            # Valores exactamente en la mediana se clasifican como Transition
            quadrants['Transition'].append((name, conf))
    
    icons = {
        'Structural Strength': 'VERDE',
        'Tactical Correction': 'AMARILLO',
        'Tactical Strength': 'AZUL',
        'Structural Weakness': 'ROJO',
        'Transition': 'GRIS'
    }
    for quadrant, sector_list in quadrants.items():
        icon = icons.get(quadrant, '?')
        if sector_list:
            sector_names = [s[0] for s in sector_list]
            avg_conf = safe_mean([s[1] for s in sector_list])
            lines.append(f"| {icon} **{quadrant}** | {', '.join(sector_names)} | {avg_conf:.0%} |\n")
        else:
            lines.append(f"| {icon} **{quadrant}** | -- | -- |\n")
    lines.append("\n")
    lines.append("*Nota: 'Structural Strength' en Opportunity Map identifica posicion relativa en el eje Structural. No implica liderazgo confirmado por SLPM.*\n\n")

    # =========================================================================
    # SLPM v1.2
    # =========================================================================
    if slpm_v12_data:
        breadth = slpm_v12_data.get('leader_breadth_v2', {})
        if breadth:
            total = breadth.get('expected_leaders', 5)
            n = breadth.get('n_used', 0)
            coverage = breadth.get('coverage', 0)
            lines.append(f"*Cobertura de lideres SLPM: {n}/{total} ({coverage:.0%})*")
            if breadth.get('coverage_warning', False):
                lines.append(" - ADVERTENCIA: Cobertura baja, resultados con incertidumbre elevada.")
            lines.append("\n\n")

        lines.append("## Structural Leadership (SLPM v1.2)\n")
        sector = slpm_v12_data.get('sector', 'N/A')
        state_v12 = slpm_v12_data.get('state', 'N/A')
        reason = slpm_v12_data.get('state_reason', '')
        quadrant = slpm_v12_data.get('opportunity_quadrant', 'N/A')
        lines.append(f"- **Sector Líder:** {sector}\n")
        lines.append("  - *Nota: El SLPM selecciona al líder combinando Structural, Breadth y Persistence. Tactical y LIS son métricas diagnósticas. No es simplemente el sector con mayor Structural Score.*\n")
        lines.append(f"- **Estado:** {state_v12}")
        if quadrant:
            lines.append(f" -> {quadrant}")
        lines.append("\n")
        if reason:
            lines.append(f"  - *{reason}*\n")
        
        inputs = slpm_v12_data.get('input_scores', {})
        if inputs:
            eff_breadth = inputs.get('effective_breadth', inputs.get('breadth', 0))
            pers_val = inputs.get('persistence')
            pers_str = f"{pers_val:.0%}" if pers_val is not None else "N/A"
            tact_val = inputs.get("tactical", 0)
            flow_val = slpm_v12_data.get("flow_divergence_v2", {}).get("composite", 0) if slpm_v12_data else 0
            struct_val = inputs.get("structural", 0)
            lis_val = slpm_v12_data.get("leader_integrity", {}).get("lis", 0) if slpm_v12_data else 0
            lines.append(f"- **Scores oficiales:** T={tact_val:+.2f} | S={struct_val:+.2f} | LIS={lis_val:+.2f} | Eff Breadth={eff_breadth:.2f} | Persist={pers_str} | LQ: P={tact_val:+.2f} C={_fmt_num(flow_val, '{:+.3f}')} S={struct_val:+.2f} Cf={lis_val:+.2f}\n")
        
        errors = slpm_v12_data.get('validation_errors', [])
        if errors:
            lines.append("\nERRORES DE VALIDACION:\n")
            for e in errors:
                lines.append(f"  - {e}\n")
        
        breadth = slpm_v12_data.get('leader_breadth_v2', {})
        if breadth:
            lines.append("\n### Leader Breadth & Health\n")
            rs_b = breadth.get('rs_breadth', 0)*100
            mom_b = breadth.get('momentum_breadth', 0)*100
            flow_b = breadth.get('flow_breadth', 0)*100
            wyck_b = breadth.get('wyckoff_breadth', 0)*100
            comp = breadth.get('composite', 0)*100
            effective = breadth.get('effective_composite', 0)*100
            n = breadth.get('n_used', 0)
            total = breadth.get('expected_leaders', 5)
            coverage = breadth.get('coverage', 0)*100
            lines.append(f"- **Leader Breadth (RS ratio > 1.0):** {rs_b:.0f}%\n")
            lines.append(f"- **Leader Momentum Breadth:** {mom_b:.0f}%\n")
            lines.append(f"- **Leader Flow Support:** {flow_b:.0f}%\n")
            lines.append(f"- **Leader Wyckoff Health:** {wyck_b:.0f}%\n")
            lines.append("  - *Scoring Wyckoff: MARKUP=1.0, ACCUMULATION=0.75, RANGE=0.0, DISTRIBUTION=-0.75, MARKDOWN=-1.0*\n")
            lines.append(f"- **Leader Health Composite (sin ajustar):** {comp:.0f}% ")
            lines.append(f"({SLPM_WEIGHTS['leader_breadth']['rs']:.2f}xRS + {SLPM_WEIGHTS['leader_breadth']['momentum']:.2f}xMom + {SLPM_WEIGHTS['leader_breadth']['flow']:.2f}xFlow + {SLPM_WEIGHTS['leader_breadth']['wyckoff']:.2f}xWyckoff)\n")
            lines.append(f"- **Effective Breadth:** {effective:.0f}% (Health Composite: {comp:.0f}%, Cobertura: {coverage:.0f}%) — Regla: si cobertura >= 50% no se aplica penalización\n")
            lines.append(f"  - N analizado: {n}/{total}\n")
            lines.append("  - *Nota: Effective Breadth = Health Composite (sin ajuste cuando cobertura >= 50%). La penalización por cobertura solo se aplica cuando la cobertura es inferior al 50%. La calidad observada (Health Composite) es independiente de la cobertura.*\n")
        
        integrity = slpm_v12_data.get('leader_integrity', {})
        if integrity:
            lis = integrity.get('lis', 0)
            n_leaders = integrity.get('n_leaders', 0)
            lines.append("\n### Leader Integrity Score (LIS)\n")
            lines.append(f"- **LIS:** {lis:+.2f} (n={n_leaders})\n")
            lines.append(f"- *Formula: LIS_individual = {SLPM_WEIGHTS['lis']['rs']:.2f}*tanh((RS-1)*2) + {SLPM_WEIGHTS['lis']['momentum']:.2f}*tanh(RS_mom*5) + {SLPM_WEIGHTS['lis']['flow']:.2f}*tanh(flow_proxy_z/2) + {SLPM_WEIGHTS['lis']['wyckoff']:.2f}*Wyckoff_score. LIS = media.*\n")
            lines.append("- *LIS mide la intensidad/calidad de la señal de los lideres, no el % que cumple condiciones (eso es el Breadth).*\n")
        
        flow_div = slpm_v12_data.get('flow_divergence_v2', {})
        if flow_div:
            lines.append("\n### Flow Divergence 2.0\n")
            lines.append(f"- **Composite:** {_fmt_num(flow_div.get('composite'), '{:+.3f}')}\n")
            lines.append(f"  - Leader vs Sector: {_fmt_num(flow_div.get('leader_flow_div'), '{:+.3f}')}\n")
            lines.append(f"  - Sector Flow vs Price: {_fmt_num(flow_div.get('sector_flow_vs_price_div'), '{:+.3f}')}\n")
            lines.append(f"  - Structural: {_fmt_num(flow_div.get('structural_flow_div'), '{:+.3f}')}\n")
            lines.append("- *Nota: Flujo medido como Flow Proxy (retorno x volumen). No implica flujo institucional real.*\n")
        lines.append("\n")

    # =========================================================================
    # LEGACY SLPM v1.0
    # =========================================================================
    if slpm_data:
        lines.append("<details>\n<summary><b>Legacy SLPM v1.0 (referencia historica)</b></summary>\n\n")
        state = slpm_data.get('state', 'N/A')
        lines.append(f"- **Sector Líder:** {slpm_data.get('sector', 'N/A')} ({slpm_data.get('sector_etf', '')})\n")
        lines.append(f"- **Estado:** {state}\n")
        lines.append(f"- **Structural RS:** {slpm_data.get('struct_rs', 0):+.3f}\n")
        lines.append(f"- **Leader Breadth:** {slpm_data.get('leader_breadth', 0)*100:.0f}%\n")
        lines.append(f"- **Flow Divergence:** {slpm_data.get('flow_divergence', 0):+.3f}\n")
        lines.append(f"- **Tactical Score (legacy):** {slpm_data.get('tactical_score', 0):+.3f}\n")
        lines.append(f"- **Structural Score (legacy):** {slpm_data.get('structural_score', 0):+.3f}\n")
        lines.append("\n</details>\n\n")

    if leader_lines:
        lines.append("\n## Acciones Seleccionadas por el Modelo de Liderazgo Sectorial\n")
        lines.append("> Solo se muestran sectores en fase ACCUMULATION o MARKUP. El resto se omiten por no cumplir criterios de liderazgo estructural.\n\n")
        lines.extend(leader_lines)
    else:
        lines.append("\n## Acciones Seleccionadas por el Modelo de Liderazgo Sectorial\n")
        lines.append("*No disponibles: ningun sector en fase de acumulación.*\n")

    # =========================================================================
    # OMS v2.0
    # =========================================================================
    if pcr_data:
        lines.append("## Sentimiento de Opciones\n")
        lines.append(f"- **PCR Total:** {pcr_data.get('total_pcr', np.nan):.2f} ")
        ewma_val = pcr_data.get('pcr_ewm', np.nan)
        if pd.notna(ewma_val):
            lines.append(f"(EWMA(5): {ewma_val:.2f})\n")
        else:
            lines.append("(EWMA(5): N/D - historial insuficiente)\n")
        if pd.notna(pcr_data.get('z_score')):
            lines.append(f"- **Robust Z-Score:** {pcr_data['z_score']:.2f}\n")
            lines.append(f"- **Momentum:** {pcr_data.get('momentum', 0):.2f}\n")
            lines.append(f"- **Percentil:** {pcr_data.get('percentile', 0):.0f}%\n")
            lines.append(f"- **Estado:** {pcr_data.get('state', 'N/A')}\n")
        lines.append(f"- **PCR Indices:** {_fmt_num(pcr_data.get('index_pcr', np.nan), '{:.2f}')} | "
                     f"**PCR Acciones:** {pcr_data.get('equity_pcr', np.nan):.2f} | "
                     f"**PCR ETP:** {pcr_data.get('etp_pcr', np.nan):.2f}\n")
        lines.append(f"- **PCR VIX:** {pcr_data.get('vix_pcr', np.nan):.2f} | "
                     f"**PCR SPX:** {pcr_data.get('spx_pcr', np.nan):.2f}\n")
        lines.append(f"- **Institutional Hedge Ratio:** {pcr_data.get('ihr', np.nan):.2f} "
                     f"({pcr_data.get('ihr_state', 'N/A')}, bandas: <1.2 Especulacion, 1.2-1.6 Equilibrado, >1.6 Cobertura institucional)\n")
        lines.append(f"- **Volumen en Indices:** {pcr_data.get('index_volume_share', np.nan):.1%} del total\n")
        lines.append(f"- **Put Share:** {pcr_data.get('put_share', np.nan):.1%} | "
                     f"**Call Share:** {pcr_data.get('call_share', np.nan):.1%}\n")
        lines.append(f"- **Volume PCR (calculado):** {pcr_data.get('volume_pcr', np.nan):.2f} | "
                     f"**OI PCR:** {pcr_data.get('oi_pcr', np.nan):.2f}\n")
        last_date = pcr_data.get('last_date', 'N/A')
        lines.append(f"- **Ultimo dato:** {last_date}")
        if last_date != 'N/A':
            try:
                data_date = pd.Timestamp(last_date)
                age = (datetime.now() - data_date).days
                lines.append(f" (desfase: {age} dias)")
            except:
                pass
        lines.append("\n")
        lines.append(f"\n*Fuente: CBOE Official Data. Timestamp: {pcr_data.get('timestamp', 'N/A')}.*\n\n")

    # =========================================================================
    # ETF PRIMARY FLOW (SPDR)
    # =========================================================================
    if etf_primary_flow_data is not None and not etf_primary_flow_data.empty:
        lines.append("## Flujo Primario ETF (SPDR)\n")
        lines.append("| Ticker | NAV | Shares Outstanding | Total Net Assets | Primary Flow $ | Flow % AUM | Flow Z |\n")
        lines.append("|--------|-----|---------------------|------------------|----------------|------------|--------|\n")
        for _, row in etf_primary_flow_data.iterrows():
            lines.append(f"| {row['ticker']} | {row['nav']:.2f} | {row['shares_outstanding']:,.0f} | {row['total_net_assets']:,.0f} | {row['primary_flow_usd']:+,.2f} | {row['primary_flow_pct']:+.2f}% | {row['primary_flow_z']:+.2f} |\n")
        lines.append(f"\n*Fuente: State Street Global Advisors (SSGA). ETF Primary Flow = ΔShares Outstanding × NAV. Z-score sobre {ETF_PRIMARY_FLOW_ZSCORE_WINDOW} sesiones.*\n\n")

    # =========================================================================
    # =========================================================================
    # SECTOR FLOW CHARACTERISTICS
    # =========================================================================
    if sector_flow_characteristics_data is not None and not sector_flow_characteristics_data.empty:
        latest_date = pd.to_datetime(sector_flow_characteristics_data['date']).max()
        flow_latest = sector_flow_characteristics_data[pd.to_datetime(sector_flow_characteristics_data['date']) == latest_date]
        lines.append("## Flujo Primario ETF — Características\n")
        lines.append("| Sector | Flujo $ | % AUM | Z | 5d Acum | 20d Acum | Pers 5d | Pers 20d | Ret 20d | Régimen |\n")
        lines.append("|--------|---------|-------|----|---------|----------|---------|----------|---------|----------|\n")
        for _, row in flow_latest.iterrows():
            regime = row.get('price_flow_regime', None)
            regime_str = regime if pd.notna(regime) else 'N/D'
            lines.append(f"| {row['sector']} | {_fmt_num(row['flow_dollar'], '{:+,.2f}')} | {_fmt_num(row['flow_pct_aum'], '{:.2f}%')} | {_fmt_num(row['flow_zscore'], '{:.2f}')} | {_fmt_num(row['flow_5d_sum'], '{:+,.2f}')} | {_fmt_num(row['flow_20d_sum'], '{:+,.2f}')} | {_fmt_num(row['persistence_5d'], '{:.0%}')} | {_fmt_num(row['persistence_20d'], '{:.0%}')} | {_fmt_num(row['price_ret_20d'], '{:.2%}')} | {regime_str} |\n")
        lines.append("\n")

    # =========================================================================
    # DIVERGENCIA PRECIO-FLUJO PRIMARIO
    # =========================================================================
    if sector_flow_characteristics_data is not None and not sector_flow_characteristics_data.empty:
        sector_flow_characteristics_data = sector_flow_characteristics_data[pd.to_datetime(sector_flow_characteristics_data['date']) == pd.to_datetime(sector_flow_characteristics_data['date']).max()]
        lines.append("## Divergencia Precio–Flujo Primario\n")
        lines.append("| Sector | Ret 5d | Flujo 5d | Régimen 5d | Ret 20d | Flujo 20d | Régimen 20d |\n")
        lines.append("|--------|--------|----------|------------|---------|-----------|-------------|\n")
        for _, row in sector_flow_characteristics_data.iterrows():
            lines.append(f"| {row['sector']} | {row['price_ret_5d']:.2%} | {row['flow_5d_sum']:+,.0f} | {row['price_flow_regime_5d']} | {row['price_ret_20d']:.2%} | {row['flow_20d_sum']:+,.0f} | {row['price_flow_regime_20d']} |\n")
        lines.append("\n")
        lines.append("*«Absorción potencial» describe una configuración de retorno negativo del precio acompañada de flujo primario acumulado positivo. Puede ser compatible con absorción, pero no confirma por sí sola absorción institucional ni establece causalidad.*\n\n")

    # =========================================================================
    # LIDERAZGO RELATIVO INTERNO
    # =========================================================================
    if rs_internal_data is not None and not rs_internal_data.empty:
        rs_internal_data = rs_internal_data[pd.to_datetime(rs_internal_data['date']) == pd.to_datetime(rs_internal_data['date']).max()]
        top_tickers = set()
        for sector in rs_internal_data['sector'].unique():
            top = rs_internal_data[rs_internal_data['sector'] == sector].nlargest(5, 'price_ret_20d')['ticker']
            top_tickers.update(top)
        report_df = rs_internal_data[rs_internal_data['ticker'].isin(top_tickers)]
        lines.append("## Liderazgo relativo interno\n")
        lines.append("| Sector | Ticker | vs mercado 20d | vs sector 20d | Clasificación |\n")
        lines.append("|--------|--------|----------------|---------------|----------------|\n")
        for _, row in report_df.iterrows():
            lines.append(f"| {row['sector']} | {row['ticker']} | {row['rs_abs_20d']:.2%} | {row['rs_internal_20d']:.2%} | {row['classification']} |\n")
        lines.append("\n")

    # =========================================================================
    # ROTACIÓN SECTORIAL RECIENTE
    # =========================================================================
    if sector_rank_deltas_data is not None and not sector_rank_deltas_data.empty:
        lines.append("## Rotación sectorial reciente\n")
        lines.append("| Sector | Rank actual | Δ5d | Δ10d | Δ20d | Lectura 5d | Lectura 10d | Lectura 20d |\n")
        lines.append("|--------|-------------|-----|------|------|------------|-------------|-------------|\n")
        for _, row in sector_rank_deltas_data.iterrows():
            lines.append(f"| {row['sector']} | {row['rank_actual']} | {row['rank_change_5d']:+.0f} | {row['rank_change_10d']:+.0f} | {row['rank_change_20d']:+.0f} | {row['lectura_5d']} | {row['lectura_10d']} | {row['lectura_20d']} |\n")
        lines.append("\n")

    # =========================================================================
    # DISPERSIÓN ENTRE SECTORES
    if sector_dispersion_data is not None and not sector_dispersion_data.empty:
        lines.append("## Dispersión entre sectores\n")
        lines.append("| Fecha | Rango (pp) | Desv (pp) | Media (pp) | Lectura | Heterogeneidad |\n")
        lines.append("|-------|------------|-----------|------------|---------|----------------|\n")
        for _, row in sector_dispersion_data.iterrows():
            date_str = pd.Timestamp(row['date']).strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/D'
            lines.append(f"| {date_str} | {row['range_pp']:.2f} | {row['std_pp']:.2f} | {row['mean_ret']:.2f} | {row['dispersion_reading']} | {row['heterogeneity_type']} |\n")
        lines.append("\n")
        lines.append("*La dispersión mide la separación entre los retornos de los 11 sectores. No es un score ni una señal.*\n\n")

    # CORRELACIÓN ENTRE SECTORES
    if sector_correlation_summary_data is not None and not sector_correlation_summary_data.empty:
        sector_correlation_summary_data = sector_correlation_summary_data[pd.to_datetime(sector_correlation_summary_data['date']) == pd.to_datetime(sector_correlation_summary_data['date']).max()]
        lines.append("## Correlación entre sectores\n")
        lines.append("| Ventana | Media | Mediana | P25 | P75 | Mín | Máx | Lectura |\n")
        lines.append("|---------|-------|---------|-----|-----|-----|-----|---------|\n")
        for _, row in sector_correlation_summary_data.iterrows():
            lines.append(f"| {int(row['window'])}d | {row['corr_mean']:.2f} | {row['corr_median']:.2f} | {row['corr_p25']:.2f} | {row['corr_p75']:.2f} | {row['corr_min']:.2f} | {row['corr_max']:.2f} | {row['correlation_reading']} |\n")
        lines.append("\n")
        lines.append("*La correlación mide el co-movimiento entre retornos sectoriales. No es un score ni una señal.*\n\n")

    # CONTEXTO CROSS-ASSET
    if cross_asset_context_data is not None and not cross_asset_context_data.empty:
        lines.append("## Contexto transversal de mercado\n")
        lines.append("| Sector | Ventana | Equity | Rates | Crédito | Commodities | FX | VIX |\n")
        lines.append("|--------|---------|--------|-------|---------|-------------|----|-----|\n")
        # Pivotar: para cada sector y ventana, extraer mean_corr por asset_class
        grouped = cross_asset_context_data.groupby(['sector','window','asset_class'])['mean_corr'].first().unstack()
        for (sector, window), row in grouped.iterrows():
            equity = row.get('equity', None)
            rates = row.get('rates', None)
            credit = row.get('credit', None)
            commodities = row.get('commodities', None)
            fx = row.get('fx', None)
            volatility = row.get('volatility', None)
            def _fmt(v):
                return f"{v:.2f}" if pd.notna(v) else "N/D"
            lines.append(f"| {sector} | {int(window)}d | {_fmt(equity)} | {_fmt(rates)} | {_fmt(credit)} | {_fmt(commodities)} | {_fmt(fx)} | {_fmt(volatility)} |\n")
        lines.append("\n")
        lines.append("*Contexto descriptivo basado en correlaciones sector-activo transversal. No implica confirmación ni causalidad.*\n\n")

    # MATRIZ DE RÉGIMEN SECTORIAL
    # =========================================================================
    if sector_regime_matrix_data is not None and not sector_regime_matrix_data.empty:
        lines.append("## Matriz de Régimen Sectorial\n")
        lines.append("| Sector | Precio 20d | % > EMA50 | Flujo 20d | Fase Wyckoff | Positivas | Lectura |\n")
        lines.append("|--------|------------|-----------|-----------|--------------|-----------|---------|\n")
        for _, row in sector_regime_matrix_data.iterrows():
            lines.append(f"| {row['sector']} | {row['price_ret_20d']:.2%} | {row['pct_above_ema50']:.1f}% | {row['flow_20d_sum']:+,.0f} | {row['wyckoff_phase']} | {row['positive_conditions']:.0f} | {row['regime_reading']} |\n")
        lines.append("\n")

    # =========================================================================
    # REPRESENTATIVIDAD DEL LÍDER
    # =========================================================================
    if leader_representativeness_data is not None and not leader_representativeness_data.empty:
        lines.append("## Representatividad del líder\n")
        lines.append("| Sector | Líder | RS ΔMed | Mom ΔMed | Flow ΔMed | WLS ΔMed | Rank pct |\n")
        lines.append("|--------|-------|---------|----------|-----------|----------|----------|\n")
        for _, row in leader_representativeness_data.iterrows():
            lines.append(f"| {row['sector']} | {row['ticker']} | {row['rs_distance_to_median']:+.4f} | {row['mom_distance_to_median']:+.4f} | {row['flow_distance_to_median']:+.2f} | {row['wls_distance_to_median']:+.2f} | {row['sector_rank_pct']:.0%} |\n")
        lines.append("\n")

    # =========================================================================
    # DISTRIBUCIÓN WYCKOFF SECTORIAL
    # =========================================================================
    if sector_wyckoff_distribution_data is not None and not sector_wyckoff_distribution_data.empty:
        lines.append("## Distribución Wyckoff sectorial\n")
        lines.append("| Sector | Acc | Markup | Range | Dist | Markdown | N | Cobertura |\n")
        lines.append("|--------|-----|--------|-------|------|----------|---|-----------|\n")
        # Mostrar solo la última fecha para evitar duplicados históricos
        df_wy = sector_wyckoff_distribution_data.copy()
        if 'date' in df_wy.columns and df_wy['date'].notna().any():
            latest = pd.to_datetime(df_wy['date']).max()
            df_wy = df_wy[pd.to_datetime(df_wy['date']) == latest]
        for _, row in df_wy.iterrows():
            lines.append(f"| {row['sector']} | {row['pct_accumulation']:.0f}% | {row['pct_markup']:.0f}% | {row['pct_range']:.0f}% | {row['pct_distribution']:.0f}% | {row['pct_markdown']:.0f}% | {row['n_valid_wyckoff']} | {row['coverage_wyckoff']:.0f}% |\n")
        lines.append("\n")

    # =========================================================================
    # DIVERGENCIA SECTOR-LÍDERES
    # =========================================================================
    if sector_leader_divergence_data is not None and not sector_leader_divergence_data.empty:
        lines.append("## Divergencia sector-líderes\n")
        lines.append("| Sector | Ret sector | Líderes + | Líderes - | Líderes > Sector | Válidos | Lectura |\n")
        lines.append("|--------|------------|-----------|-----------|------------------|---------|---------|\n")
        for _, row in sector_leader_divergence_data.iterrows():
            lines.append(f"| {row['sector']} | {row['sector_ret_20d']:.2%} | {row['n_leaders_positive']} | {row['n_leaders_negative']} | {row['n_leaders_beating_sector']} | {row['n_leaders_valid']} | {row['classification']} |\n")
        lines.append("\n")

    # =========================================================================
    # MOMENTUM DE AMPLITUD
    # =========================================================================
    if sector_breadth_momentum_data is not None and not sector_breadth_momentum_data.empty:
        lines.append("## Momentum de amplitud\n")
        lines.append("| Sector | Δ1d EMA20 | Δ5d EMA20 | Δ20d EMA20 | Δ5d EMA50 | Δ5d EMA200 | Expansión | Deterioro |\n")
        lines.append("|--------|-----------|-----------|------------|-----------|------------|-----------|-----------|\n")
        # Mostrar solo la última fecha para evitar duplicados históricos
        df_mom = sector_breadth_momentum_data.copy()
        if 'date' in df_mom.columns and df_mom['date'].notna().any():
            latest = pd.to_datetime(df_mom['date']).max()
            df_mom = df_mom[pd.to_datetime(df_mom['date']) == latest]
        for _, row in df_mom.iterrows():
            lines.append(f"| {row['sector']} | {row['delta_1d_ema20']:+.1f} | {row['delta_5d_ema20']:+.1f} | {row['delta_20d_ema20']:+.1f} | {row['delta_5d_ema50']:+.1f} | {row['delta_5d_ema200']:+.1f} | {row['breadth_expansion_5d']} | {row['breadth_deterioration_5d']} |\n")
        lines.append("\n")

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

