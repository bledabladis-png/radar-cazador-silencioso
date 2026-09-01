import pandas as pd
import numpy as np
import os
from datetime import datetime
from config.tickers import SECTOR_NAMES
from config.index_tickers import INDEX_CONFIG
from config.settings import MOMENTUM_PRICE_WINDOW, MOMENTUM_LONG_WINDOW, ETF_PRIMARY_FLOW_ZSCORE_WINDOW
from config.weights import SLPM_WEIGHTS

MODEL_VERSION = "4.3"
WEIGHTS_VERSION = "3"
INDICATORS_VERSION = "2"

def _classify_freshness(age_days, max_current=3, max_recent=7, max_stale=14):
    if age_days <= max_current:
        return 'CURRENT'
    elif age_days <= max_recent:
        return 'RECENT'
    elif age_days <= max_stale:
        return 'STALE'
    return 'ARCHIVAL'

def _classify_finra_freshness(age_days):
    """Clasificación de frescura para FINRA Dark Pools.
    Retraso regulatorio: 2-4 semanas (14-30 días). Umbrales amplios."""
    if age_days <= 30:
        return 'CURRENT'
    elif age_days <= 45:
        return 'RECENT'
    elif age_days <= 60:
        return 'STALE'
    return 'ARCHIVAL'


def _generate_coverage_table(pcr_data, darkpool_data, sector_results):
    lines = []
    lines.append("### Cobertura de Datos\n")
    lines.append("| Fuente | Cobertura | Antigüedad |\n")
    lines.append("|--------|-----------|------------|\n")
    sectores_total = 11
    sectores_validos = sectores_total
    if sector_results and 'ranking' in sector_results:
        sectores_validos = len([s for s in sector_results['ranking'] if s[1] is not None])
    lines.append(f"| Sectores | {sectores_validos}/{sectores_total} ({sectores_validos/sectores_total:.0%}) | - |\n")
    n_acciones = 110
    try:
        import pandas as pd
        df = pd.read_csv('outputs/report/analisis_lideres.csv')
        if 'ticker' in df.columns:
            n_acciones = len(df['ticker'].unique())
    except:
        pass
    lines.append(f"| Acciones lideres | {n_acciones} tickers | - |\n")
    if pcr_data and pcr_data.get('last_date'):
        from datetime import datetime
        import pandas as pd
        pcr_age = (datetime.now() - pd.Timestamp(pcr_data['last_date'])).days
        lines.append(f"| Opciones (CBOE) | - | {pcr_age} dias |\n")
    else:
        lines.append("| Opciones (CBOE) | - | Sin datos |\n")
    if darkpool_data and darkpool_data.get('week'):
        from datetime import datetime
        import pandas as pd
        dp_age = (datetime.now() - pd.Timestamp(darkpool_data['week'])).days
        lines.append(f"| Dark Pool (FINRA) | - | {dp_age} dias |\n")
    else:
        lines.append("| Dark Pool (FINRA) | - | Sin datos |\n")
    lines.append("\n")
    return lines


def generate_daily_report(macro_score, macro_regime, macro_conf, liquidity_score, liquidity_regime, liq_conf,
                          volatility_score, vol_regime, vol_conf, sector_results,
                          sector_price_rank, sector_flow_rank, otros_price_rank, otros_flow_rank,
                          leader_lines=None, breadth_values=None, real_liquidity_regime=None, real_liquidity_conf=None,
                          pcr_data=None, darkpool_data=None, mte_result=None, confirmation_data=None, slpm_data=None,
                          slpm_v12_data=None, tactical_scores=None, structural_scores=None,
                          sector_persistence=None, signal_agreements=None, signal_agreements_display=None,
                          cross_module_conflict=None, shock_sensitivities=None, price_flow_divergences=None,
                          dc_summary="", all_signals=None, real_liq_score=None, real_liq_prev=None, index_leaders=None, index_phases=None, etf_primary_flow_data=None, cftc_position_flow_data=None, flow_synthesis=None, blackrock_dax_flow=None, blackrock_isf_flow=None, amundi_lyxi_flow=None, blackrock_iwm_flow=None, nport_position_change_data=None, qqq_performance_data=None, qqq_nport_flow_data=None, qqq_sec_flow=None, output_path='outputs/report/reporte_diario.md'):
    lines = []
    lines.append("# MACRO SECTORIAL - Reporte Diario\n")
    lines.append(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Modelo:** v{MODEL_VERSION} | Pesos: v{WEIGHTS_VERSION} | Indicadores: v{INDICATORS_VERSION}\n\n")

    # =========================================================================
    # RESUMEN DE REGIMENES
    # =========================================================================
    lines.append("## Resumen de Regimenes\n")
    try:
        score_value = float(macro_score.iloc[-1])
    except Exception:
        score_value = float('nan')

    if pd.isna(score_value):
        score_str = "N/D"
    else:
        score_str = f"{score_value:.2f}"

    lines.append(f"- **Macro:** {macro_regime} (Score: {score_str}, Signal Consistency: {macro_conf:.0%})\n")
    if macro_conf < 0.30:
        lines.append("  *Signal Consistency baja: señales contradictorias en el entorno actual.*\n")
    
    lines.append(f"- **Cond. Financieras:** {liquidity_regime} (Score: {liquidity_score.iloc[-1]:.2f}, Signal Consistency: {liq_conf:.0%})\n")
    if liquidity_regime == 'HIGH_STRESS':
        lines.append("  *Nota: El modulo financiero detecta estres significativo, pero volatilidad y liquidez no confirman un deterioro transversal. No se clasifica como CRISIS sistemica.*\n")
    
    if real_liquidity_regime is not None:
        lines.append(f"- **Liquidez Real (FRED):** {real_liquidity_regime} (Signal Consistency: {real_liquidity_conf:.0%})\n")
    if real_liq_prev is not None:
        try:
            delta = float(real_liq_score.iloc[-1]) - float(real_liq_prev)
            if delta > 0.05:
                delta_str = "MEJORA"
            elif delta < -0.05:
                delta_str = "EMPEORA"
            else:
                delta_str = "ESTABLE"
            lines.append(f"  - *Liquidity Delta (vs ejecución anterior): {delta:+.3f} ({delta_str})*\n")
        except:
            pass
    
    vol_z = volatility_score.iloc[-1] if hasattr(volatility_score, 'iloc') else volatility_score
    if vol_conf < 0.05 and abs(vol_z) < 0.1:
        vol_conf_str = "Señal neutra (sin desviación significativa)"
    else:
        vol_conf_str = f"Signal Consistency: {vol_conf:.0%}"
    vol_z_display = "0.00" if abs(vol_z) < 0.005 else f"{vol_z:.2f}"
    lines.append(f"- **Volatilidad:** {vol_regime} (Z-Score: {vol_z_display}, {vol_conf_str})\n")

    sector_regime = sector_results['regime']
    lines.append(f"- **Sectores:** {sector_regime}\n")
    lines.append("*Nota: Signal Consistency mide la consistencia entre señales, no una probabilidad estadistica calibrada. Data Conf mide la frescura y cobertura de los datos.*\n\n")

    # =========================================================================
    # DATA FRESHNESS
    # =========================================================================
    lines.append("### Data Freshness\n")
    lines.append("| Fuente | Ultimo dato | Antigüedad | Estado | Data Conf |\n")
    lines.append("|--------|-------------|------------|--------|----------|\n")
    now = datetime.now()
    
    if pcr_data and pcr_data.get('last_date', 'N/A') != 'N/A':
        try:
            d = pd.Timestamp(pcr_data['last_date'])
            age = (now - d).days
            cboe_status = _classify_freshness(age, 3, 5, 10)
            cboe_conf = 'Alta' if cboe_status in ('CURRENT', 'RECENT') else 'Baja'
            lines.append(f"| CBOE (Opciones) | {d.strftime('%Y-%m-%d')} | {age} dias | {cboe_status} | {cboe_conf} |\n")
        except:
            lines.append(f"| CBOE (Opciones) | {pcr_data.get('last_date', 'N/A')} | N/D | N/D | N/D |\n")
    else:
        lines.append("| CBOE (Opciones) | N/D | N/D | N/D | N/D |\n")
    
    if darkpool_data:
        week = darkpool_data.get('week', 'N/A')
        if week != 'N/A':
            try:
                d = pd.Timestamp(week)
                age = (now - d).days
                finra_status = _classify_finra_freshness(age)
                finra_conf = 'Alta' if finra_status in ('CURRENT', 'RECENT') else 'Baja'
                lines.append(f"| FINRA (Dark Pools) | {d.strftime('%Y-%m-%d')} | {age} dias | {finra_status} | {finra_conf} |\n")
            except:
                lines.append(f"| FINRA (Dark Pools) | {week} | N/D | N/D | N/D |\n")
        else:
            lines.append("| FINRA (Dark Pools) | N/D | N/D | N/D | N/D |\n")
    
    # FRED: intentar obtener fecha real desde liquidity_state.json
    try:
        import json
        from pathlib import Path as _Path
        liq_state_path = _Path('outputs/state/liquidity_state.json')
        if liq_state_path.exists():
            with liq_state_path.open('r') as f:
                liq_state = json.load(f)
            last_fred = liq_state.get('date', 'N/A')
            if last_fred != 'N/A':
                d = pd.Timestamp(last_fred)
                age = (now - d).days
                fred_status = _classify_freshness(age, 7, 14, 21)
                fred_conf = 'Alta' if fred_status in ('CURRENT', 'RECENT') else 'Baja'
                lines.append(f"| FRED (Macro) | {d.strftime('%Y-%m-%d')} | {age} dias | {fred_status} | {fred_conf} |\n")
            else:
                lines.append("| FRED (Macro) | N/D | N/D | N/D | N/D |\n")
        else:
            lines.append("| FRED (Macro) | N/D | N/D | N/D | N/D |\n")
    except Exception:
        lines.append("| FRED (Macro) | N/D | N/D | N/D | N/D |\n")
    lines.append("| Yahoo Finance (Precios) | Diario | < 1 dia | CURRENT | Alta |\n")
    lines.append("\n")
    coverage_lines = _generate_coverage_table(pcr_data, darkpool_data, sector_results)
    for cl in coverage_lines:
        lines.append(cl)


    # =========================================================================
    # ALERTAS DE DIVERGENCIA (CORREGIDAS - sin flujo institucional ni shock externo)
    # =========================================================================
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
        lines.append("### Alertas de Divergencia (Inicial)\n")
        for alert in alerts:
            lines.append(alert + "\n")
        lines.append("\n")
    else:
        lines.append("### Alertas de Divergencia (Inicial)\n")
        lines.append("*Sin divergencias relevantes.*\n\n")

    # =========================================================================
    # CROSS-MODULE CONFLICT
    # =========================================================================
    if cross_module_conflict:
        level = cross_module_conflict.get('conflict_level', 'MIXED')
        icon = 'OK' if level == 'CONSENSUS' else 'WARN' if level in ('CONFLICT', 'DIVERGENCE') else 'INFO'
        lines.append(f"### {icon} Cross-Module: {level}\n")
        lines.append(f"**Mensaje:** {cross_module_conflict.get('message', '')}\n")
        blocks = cross_module_conflict.get('blocks', '')
        if blocks:
            lines.append(f"**Bloques:** {blocks}\n")
        details = cross_module_conflict.get('details', {})
        if details:
            lines.append("\n**Detalle por modulo:**\n")
            for mod_name, mod_info in details.items():
                state = mod_info.get('state', 'N/A')
                bias_fin = mod_info.get('bias_financial', 0)
                bias_inf = mod_info.get('bias_inflation', 0)
                bias_str = ''
                if bias_fin == -1: bias_str += 'Estres Financiero '
                if bias_inf == -1: bias_str += 'Presión Inflacionaria '
                if bias_str == '': bias_str = 'Neutral'
                lines.append(f"- {mod_name}: {state} ({bias_str})\n")
        lines.append("\n")

    if breadth_values:
        lines.append("## Breadth de Mercado (11 sectores)\n")
        lines.append("| Metrica | Valor |\n")
        lines.append("|---------|-------|\n")
        b20_pct = breadth_values.get('% sobre EMA20', 0)
        b50_pct = breadth_values.get('% sobre EMA50', 0)
        b200_pct = breadth_values.get('% sobre EMA200', 0)
        nh_pct = breadth_values.get('New Highs (%)', 0)
        nl_pct = breadth_values.get('New Lows (%)', 0)

        b20_count = breadth_values.get('EMA20 count', int(round(b20_pct * 11)))
        b50_count = breadth_values.get('EMA50 count', int(round(b50_pct * 11)))
        b200_count = breadth_values.get('EMA200 count', int(round(b200_pct * 11)))
        nh_count = breadth_values.get('New Highs count', int(round(nh_pct * 11)))
        nl_count = breadth_values.get('New Lows count', int(round(nl_pct * 11)))

        lines.append(f"| % sobre EMA20 | {b20_count}/11 ({b20_pct:.2%}) |\n")
        lines.append(f"| % sobre EMA50 | {b50_count}/11 ({b50_pct:.2%}) |\n")
        lines.append(f"| % sobre EMA200 | {b200_count}/11 ({b200_pct:.2%}) |\n")
        lines.append(f"| New Highs sectoriales | {nh_count}/11 ({nh_pct:.2%}) |\n")
        lines.append(f"| New Lows sectoriales | {nl_count}/11 ({nl_pct:.2%}) |\n")
        lines.append("\n")

    # =========================================================================
    # TACTICAL LEADERS
    # =========================================================================
    lines.append(f"\n## Momentum de Precio - Sectores ({MOMENTUM_PRICE_WINDOW} dias)\n")
    lines.append("| # | Sector | Retorno 20d (%) |\n")
    lines.append("|---|--------|------------------|\n")
    for i, (ticker, mom) in enumerate(sector_price_rank[:11], 1):
        name = SECTOR_NAMES.get(ticker, ticker)
        lines.append(f"| {i} | {name} ({ticker}) | {mom*100:.2f}% |\n")

    lines.append("\n## Flujo Institucional - Sectores (Proxy)\n")
    lines.append("| # | Sector | Flujo (z-score) |\n")
    lines.append("|---|--------|------------------|\n")
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
        flow = next((f for t, f in sector_flow_rank if t == ticker), 0)
        shock = shock_sensitivities.get(ticker, {}) if shock_sensitivities else {}
        comm = shock.get('commodity_level', 'N/A') if shock else 'N/A'
        comm_val = shock.get('commodity_corr_value', None) if shock else None
        comm_display = f"{comm} ({comm_val:+.2f})" if comm_val is not None and comm != 'N/A' else comm
        lines.append(f"| {i} | {name} ({ticker}) | {t_score:+.2f} | {s_score:+.2f} | {mom*100:.2f}% | {flow:+.2f} | {comm_display} |\n")
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

    lines.append("\n## Flujo Institucional - Otros Activos (Proxy)\n")
    lines.append("| # | Activo | Flujo (z-score) |\n")
    lines.append("|---|--------|------------------|\n")
    for i, (ticker, flow) in enumerate(otros_flow_rank[:15], 1):
        lines.append(f"| {i} | {ticker} | {flow:.2f} |\n")

    lines.append("## Structural Ranking (Fortaleza de largo plazo)\n")
    lines.append("| # | Sector | Structural | Tactical | Persist | Agreement | Signal Consistency |\n")
    lines.append("|---|--------|------------|----------|---------|-----------|------------|\n")
    structural_ranking = sorted(structural_scores.items(), key=lambda x: x[1], reverse=True) if structural_scores else []
    for i, (ticker, s_score) in enumerate(structural_ranking[:11], 1):
        name = SECTOR_NAMES.get(ticker, ticker)
        t_score = tactical_scores.get(ticker, 0.0) if tactical_scores else 0.0
        pers_raw = sector_persistence.get(ticker) if sector_persistence else None; pers = pers_raw if pers_raw is not None else "N/A"
        agree = signal_agreements.get(ticker, 0.5) if signal_agreements else 0.5
        agree_display = signal_agreements_display.get(ticker, f'{agree:.0%}') if signal_agreements_display else f'{agree:.0%}'
        struct_conf = (pers + agree) / 2
        lines.append(f"| {i} | {name} ({ticker}) | {s_score:+.2f} | {t_score:+.2f} | {pers:.0%} | {agree_display} | {struct_conf:.0%} |\n")
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
        pers_raw = sector_persistence.get(ticker) if sector_persistence else None; pers = pers_raw if pers_raw is not None else "N/A"
        agree = signal_agreements.get(ticker, 0.5) if signal_agreements else 0.5
        agree_display = signal_agreements_display.get(ticker, f'{agree:.0%}') if signal_agreements_display else f'{agree:.0%}'
        shock = shock_sensitivities.get(ticker, {}) if shock_sensitivities else {}
        comm_level = shock.get('commodity_level', 'N/A') if shock else 'N/A'
        comm_val = shock.get('commodity_corr_value', None) if shock else None
        comm_display = f"{comm_level} ({comm_val:+.2f})" if comm_val is not None and comm_level != 'N/A' else comm_level
        lines.append(f"| {i} | {name} ({ticker}) | {score:.2f} | {t_score:+.2f} | {s_score:+.2f} | {pers:.0%} | {agree_display} | {comm_display} | {wyckoff} |\n")
    lines.append("\n")

    # =========================================================================
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
        pers_raw = sector_persistence.get(ticker) if sector_persistence else None; pers = pers_raw if pers_raw is not None else "N/A"
        agree = signal_agreements.get(ticker, 0.5) if signal_agreements else 0.5
        conf = (pers + agree) / 2
        
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
            avg_conf = np.mean([s[1] for s in sector_list])
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
        lines.append(f"- **Sector Lider:** {sector}\n")
        lines.append("  - *Nota: El SLPM selecciona al lider combinando Structural, Breadth y Persistence. Tactical y LIS son métricas diagnósticas. No es simplemente el sector con mayor Structural Score.*\n")
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
            lines.append(f"- **Scores oficiales:** T={tact_val:+.2f} | S={struct_val:+.2f} | LIS={lis_val:+.2f} | Eff Breadth={eff_breadth:.2f} | Persist={pers_str} | LQ: P={tact_val:+.2f} C={flow_val:+.3f} S={struct_val:+.2f} Cf={lis_val:+.2f}\n")
        
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
            lines.append("({SLPM_WEIGHTS['leader_breadth']['rs']:.2f}xRS + {SLPM_WEIGHTS['leader_breadth']['momentum']:.2f}xMom + {SLPM_WEIGHTS['leader_breadth']['flow']:.2f}xFlow + {SLPM_WEIGHTS['leader_breadth']['wyckoff']:.2f}xWyckoff)\n")
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
            lines.append(f"- **Composite:** {flow_div.get('composite', 0):+.3f}\n")
            lines.append(f"  - Leader vs Sector: {flow_div.get('leader_flow_div', 0):+.3f}\n")
            lines.append(f"  - Sector Flow vs Price: {flow_div.get('sector_flow_vs_price_div', 0):+.3f}\n")
            lines.append(f"  - Structural: {flow_div.get('structural_flow_div', 0):+.3f}\n")
            lines.append("- *Nota: Flujo medido como Flow Proxy (retorno x volumen). No implica flujo institucional real.*\n")
        lines.append("\n")

    # =========================================================================
    # LEGACY SLPM v1.0
    # =========================================================================
    if slpm_data:
        lines.append("<details>\n<summary><b>Legacy SLPM v1.0 (referencia historica)</b></summary>\n\n")
        state = slpm_data.get('state', 'N/A')
        lines.append(f"- **Sector Lider:** {slpm_data.get('sector', 'N/A')} ({slpm_data.get('sector_etf', '')})\n")
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
        lines.append(f"- **PCR Indices:** {pcr_data.get('index_pcr', np.nan):.2f} | "
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
            lines.append(f"| {row['ticker']} | {row['nav']:.2f} | {row['shares_outstanding']:,.0f} | {row['total_net_assets']:,.0f} | {row['primary_flow_usd']:+,.0f} | {row['primary_flow_pct']:+.2f}% | {row['primary_flow_z']:+.2f} |\n")
        lines.append(f"\n*Fuente: State Street Global Advisors (SSGA). ETF Primary Flow = ΔShares Outstanding × NAV. Z-score sobre {ETF_PRIMARY_FLOW_ZSCORE_WINDOW} sesiones.*\n\n")

    # =========================================================================
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
                lines.append(f"*Fecha de calculo (as_of_date): {as_of}*\n")
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
        lines.append(f"| Flow Proxy | {flow_synthesis.get('flow_proxy_sign', 0):+.2f} |\n")
        lines.append(f"| ETF Primary Flow | {flow_synthesis.get('etf_primary_flow_sign', 0):+.2f} |\n")
        lines.append(f"| CFTC Position Flow | {flow_synthesis.get('cftc_flow_sign', 0):+.2f} |\n")
        lines.append(f"| Europa Primary Flow | {flow_synthesis.get('european_flow_sign', 0):+.2f} |\n")
        lines.append(f"\n**FLOW_CONFIDENCE:** {flow_synthesis.get('confidence', 'N/A')}\n")
        lines.append("\n*Interpretación descriptiva: concordancia de signos entre capas. No es señal predictiva.*\n\n")

    # =========================================================================
    # MTE v1.0
    # =========================================================================
    if mte_result:
        lines.append("## Market Transition Engine (MTE v1.0)\n")
        mte_conf = mte_result.get('confidence', 0)
        mte_scenario = mte_result.get('scenario', 'N/A')
        if mte_conf < 0.5:
            lines.append(f"- **Escenario (UNCONFIRMED):** {mte_scenario} (Confidence Score no calibrado: {mte_conf:.2f}) - *No se considera confirmado.*\n")
        else:
            lines.append(f"- **Escenario:** {mte_scenario} (Confidence Score no calibrado: {mte_conf:.2f})\n")
        lines.append("*Nota: Confidence Score (no calibrado, escala 0-1) representa la distancia a los umbrales y el consenso entre motores. No debe interpretarse como probabilidad.*\n")
        lines.append(f"- **Market Stress Index (MSI):** {mte_result.get('msi', 0):.0f}\n")
        lines.append(f"- **Inflation Pressure Index (IPI):** {mte_result.get('ipi', 0):.0f}\n")
        lines.append(f"- **Sector Rotation Score:** {mte_result.get('srs', 0):.2f}\n")
        lines.append(f"- **Safe Haven Score:** {mte_result.get('shs', 0):.2f}\n")
        lines.append(f"- **Credit Stress Score:** {mte_result.get('cls', 0):.2f}")
        lines.append(" (orientacion: positivo = mayor estres crediticio)\n")
        lines.append(f"- **Inflation Pressure Score:** {mte_result.get('ips', 0):.2f}\n\n")

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
            lines.append(f"- **Realized Vol (21d):** {confirmation_data['rv_21d']*100:.2f}%\n")
        if confirmation_data.get('rv_60d') is not None:
            lines.append(f"- **Realized Vol (60d):** {confirmation_data['rv_60d']*100:.2f}%\n")
        if confirmation_data.get('vrp_21d') is not None:
            lines.append(f"- **VRP Proxy (VIX - RV21):** {confirmation_data['vrp_21d']*100:+.2f}%\n")
        if confirmation_data.get('vrp_60d') is not None:
            lines.append(f"- **VRP Proxy (VIX - RV60):** {confirmation_data['vrp_60d']*100:+.2f}%\n")

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
        resumen.append("- **Régimen macro: MIXED** — ROTATIONAL / MIXED — rotación sectorial activa con dispersión elevada.")
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








