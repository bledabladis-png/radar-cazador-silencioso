import pandas as pd
import numpy as np
import os
from datetime import datetime
from config.tickers import SECTOR_NAMES

MODEL_VERSION = "3.1"
WEIGHTS_VERSION = "3"
INDICATORS_VERSION = "2"

def generate_daily_report(macro_score, macro_regime, macro_conf, liquidity_score, liquidity_regime, liq_conf,
                          volatility_score, vol_regime, vol_conf, sector_results,
                          price_ranking, flow_ranking,
                          leader_lines=None, breadth_values=None, real_liquidity_regime=None, real_liquidity_conf=None,pcr_data=None, darkpool_data=None,
                          output_path='outputs/reporte_diario.md'):
    lines = []
    lines.append("# MACRO SECTORIAL - Reporte Diario\n")
    lines.append(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Modelo:** v{MODEL_VERSION} | Pesos: v{WEIGHTS_VERSION} | Indicadores: v{INDICATORS_VERSION}\n\n")

    lines.append("## Resumen de Regimenes\n")
    lines.append(f"- **Macro:** {macro_regime} (Score: {macro_score.iloc[-1]:.2f}, Confianza: {macro_conf:.0%})\n")
    if macro_conf < 0.30:
        lines.append("  *Confianza baja: senhales contradictorias en el entorno actual.*\n")
    lines.append(f"- **Cond. Financieras:** {liquidity_regime} (Score: {liquidity_score.iloc[-1]:.2f}, Confianza: {liq_conf:.0%})\n")
    if real_liquidity_regime is not None:
        lines.append(f"- **Liquidez Real (FRED):** {real_liquidity_regime} (Confianza: {real_liquidity_conf:.0%})\n")
    lines.append(f"- **Volatilidad:** {vol_regime} (Z-Score: {volatility_score.iloc[-1]:.2f}, Confianza: {vol_conf:.0%})\n")

    sector_regime = sector_results['regime']
    lines.append(f"- **Sectores:** {sector_regime}\n\n")

    if breadth_values:
        lines.append("## Breadth de Mercado (11 sectores)\n")
        lines.append("| Metrica | Valor |\n")
        lines.append("|---------|-------|\n")
        for name, val in breadth_values.items():
            lines.append(f"| {name} | {val:.2%} |\n")
        lines.append("\n")

    lines.append("## Rankings Sectoriales\n")
    lines.append("| # | Sector | Score | Fase Wyckoff |\n")
    lines.append("|---|--------|-------|---------------|\n")
    for i, (ticker, name, score, wyckoff) in enumerate(sector_results['ranking'][:11], 1):
        lines.append(f"| {i} | {name} ({ticker}) | {score:.2f} | {wyckoff} |\n")

    # Separar sectores del resto
    sector_tickers = list(SECTOR_NAMES.keys())
    sector_price = [(t, m) for t, m in price_ranking if t in sector_tickers]
    sector_flow = [(t, f) for t, f in flow_ranking if t in sector_tickers]
    otros_price = [(t, m) for t, m in price_ranking if t not in sector_tickers]
    otros_flow = [(t, f) for t, f in flow_ranking if t not in sector_tickers]

    lines.append("\n## Momentum de Precio - Sectores (20 dias)\n")
    lines.append("| # | Sector | Retorno 20d (%) |\n")
    lines.append("|---|--------|------------------|\n")
    for i, (ticker, mom) in enumerate(sector_price, 1):
        name = SECTOR_NAMES.get(ticker, ticker)
        lines.append(f"| {i} | {name} ({ticker}) | {mom*100:.2f}% |\n")

    lines.append("\n## Flujo Institucional - Sectores (Proxy)\n")
    lines.append("| # | Sector | Flujo (z-score) |\n")
    lines.append("|---|--------|------------------|\n")
    for i, (ticker, flow) in enumerate(sector_flow, 1):
        name = SECTOR_NAMES.get(ticker, ticker)
        lines.append(f"| {i} | {name} ({ticker}) | {flow:.2f} |\n")

    lines.append("\n## Momentum de Precio - Otros Activos (20 dias)\n")
    lines.append("| # | Activo | Retorno 20d (%) |\n")
    lines.append("|---|--------|------------------|\n")
    for i, (ticker, mom) in enumerate(otros_price[:20], 1):
        lines.append(f"| {i} | {ticker} | {mom*100:.2f}% |\n")

    lines.append("\n## Flujo Institucional - Otros Activos (Proxy)\n")
    lines.append("| # | Activo | Flujo (z-score) |\n")
    lines.append("|---|--------|------------------|\n")
    for i, (ticker, flow) in enumerate(otros_flow[:20], 1):
        lines.append(f"| {i} | {ticker} | {flow:.2f} |\n")

    if leader_lines:
        lines.append("\n## Lideres Sectoriales\n")
        lines.append("> Acciones con mejor perfil institucional dentro de sectores favorables.\n\n")
        lines.extend(leader_lines)
    else:
        lines.append("\n## Lideres Sectoriales\n")
        lines.append("*No disponibles: ningun sector en fase de acumulacion.*\n")

    # --- Sentimiento de Opciones (PCR) ---
    if pcr_data:
        lines.append("## Sentimiento de Opciones (OMS v1.1)\n")
        lines.append(f"- **OMS STATUS:** {pcr_data.get('status', 'N/A')}\n")
        if pcr_data.get('issues'):
            for issue in pcr_data.get('issues', []):
                lines.append(f"  - ⚠️ {issue}\n")
        if 'pcr_total' in pcr_data:
            lines.append(f"- **PCR Total:** {pcr_data.get('pcr_total', np.nan):.2f} "
                         f"(Z-Score: {pcr_data.get('z_score', np.nan):.2f} | "
                         f"Percentil 3Y: {pcr_data.get('percentile_3y', np.nan):.0%} | "
                         f"Percentil 10Y: {pcr_data.get('percentile_10y', np.nan):.0%})\n")
        if 'pcr_equity' in pcr_data or 'pcr_index' in pcr_data:
            lines.append(f"- **PCR Acciones:** {pcr_data.get('pcr_equity', np.nan):.2f} | "
                         f"**PCR Índices:** {pcr_data.get('pcr_index', np.nan):.2f}\n")
        if pcr_data.get('divergence_flag') and pcr_data.get('divergence_flag') != 'No divergence':
            lines.append(f"- **Divergence Flag:** {pcr_data['divergence_flag']}\n")
        if 'extreme_flag' in pcr_data:
            lines.append(f"- **Extreme Flag:** {pcr_data.get('extreme_flag', 'N/A')}\n")
        if 'lectura_contrarian' in pcr_data:
            lines.append(f"- **Lectura contrarian:** {pcr_data.get('lectura_contrarian', 'N/A')}\n")
        if 'days_since' in pcr_data:
            lines.append(f"- **Data Freshness:** Ultimo dato {pcr_data.get('last_date', 'N/A')} "
                         f"({pcr_data.get('days_since', '?')} dias de retraso)\n")
        if 'coverage' in pcr_data:
            lines.append(f"- **Cobertura:** {pcr_data.get('coverage', 0):.0%} "
                         f"({len(pcr_data.get('available_series', []))}/{len(pcr_data.get('required_series', []))} series)\n")
        if pcr_data.get('vix_correlation') is not None:
            lines.append(f"- **Correlación con VIX (252d):** {pcr_data['vix_correlation']:.2f}\n")
        lines.append(f"\n*Fuente: FRED. Timestamp: {pcr_data.get('timestamp', 'N/A')}.*\n\n")

    # --- Dark Pools (FINRA ATS) ---
    if darkpool_data:
        lines.append("## Flujos Institucionales (Dark Pools v1.0)\n")
        lines.append(f"- **Dark Pool medio:** {darkpool_data.get('media_dark_pool', 0):.2f}% "
                     f"({darkpool_data.get('n_tickers_ats', 0)}/{darkpool_data.get('n_tickers_total', 0)} tickers)\n")
        lines.append(f"- **Mayor Dark Pool %:** {darkpool_data.get('ticker_max', 'N/A')} "
                     f"({darkpool_data.get('max_dark_pool', 0):.2f}%)\n")
        lines.append(f"- **Semana FINRA:** {darkpool_data.get('week', 'N/A')}\n")
        lines.append(f"\n*Fuente: FINRA ATS Transparency Data.*\n\n")

    lines.append("\n---\n")
    lines.append("*Informe generado automaticamente por Macro Sectorial v3.1. No constituye recomendacion de inversion.*\n")
    lines.append("*El sistema implementa un conjunto consistente de reglas deterministas, con normalizacion robusta, separacion modular y metodologia documentada.*\n")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    # Guardar historico de regimenes
    hist_path = 'outputs/macro_regime.csv'
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

    # Guardar rankings sectoriales
    sector_df = pd.DataFrame(sector_results['ranking'], columns=['ticker', 'name', 'score', 'wyckoff_phase'])
    sector_df.to_csv('outputs/sector_rankings.csv', index=False)
