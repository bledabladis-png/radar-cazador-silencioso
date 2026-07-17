import pandas as pd
import numpy as np
import os
from datetime import datetime
from config.tickers import SECTOR_NAMES

MODEL_VERSION = "3.15"
WEIGHTS_VERSION = "3"
INDICATORS_VERSION = "2"

def generate_daily_report(macro_score, macro_regime, macro_conf, liquidity_score, liquidity_regime, liq_conf,
                          volatility_score, vol_regime, vol_conf, sector_results,
                          price_ranking, flow_ranking,
                          leader_lines=None, breadth_values=None, real_liquidity_regime=None, real_liquidity_conf=None,
                          pcr_data=None, darkpool_data=None, mte_result=None,
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

    # --- Sentimiento de Opciones (OMS v2.0) ---
    if pcr_data:
        lines.append("## Sentimiento de Opciones (OMS v2.0)\n")
        lines.append(f"- **PCR Total:** {pcr_data.get('total_pcr', np.nan):.2f} "
                     f"(EWMA(5): {pcr_data.get('pcr_ewm', np.nan):.2f})\n")
        if pd.notna(pcr_data.get('z_score')):
            lines.append(f"- **Robust Z-Score:** {pcr_data['z_score']:.2f}\n")
            lines.append(f"- **Momentum:** {pcr_data.get('momentum', 0):.2f}\n")
            lines.append(f"- **Percentil:** {pcr_data.get('percentile', 0):.0f}%\n")
            lines.append(f"- **Estado:** {pcr_data.get('state', 'N/A')}\n")
        lines.append(f"- **PCR Índices:** {pcr_data.get('index_pcr', np.nan):.2f} | "
                     f"**PCR Acciones:** {pcr_data.get('equity_pcr', np.nan):.2f} | "
                     f"**PCR ETP:** {pcr_data.get('etp_pcr', np.nan):.2f}\n")
        lines.append(f"- **PCR VIX:** {pcr_data.get('vix_pcr', np.nan):.2f} | "
                     f"**PCR SPX:** {pcr_data.get('spx_pcr', np.nan):.2f}\n")
        # Nuevos indicadores v2.0
        lines.append(f"- **Institutional Hedge Ratio:** {pcr_data.get('ihr', np.nan):.2f} "
                     f"({pcr_data.get('ihr_state', 'N/A')})\n")
        lines.append(f"- **Volumen en Índices:** {pcr_data.get('index_volume_share', np.nan):.1%} del total\n")
        lines.append(f"- **Put Share:** {pcr_data.get('put_share', np.nan):.1%} | "
                     f"**Call Share:** {pcr_data.get('call_share', np.nan):.1%}\n")
        lines.append(f"- **Volume PCR (calculado):** {pcr_data.get('volume_pcr', np.nan):.2f} | "
                     f"**OI PCR:** {pcr_data.get('oi_pcr', np.nan):.2f}\n")
        lines.append(f"- **Último dato:** {pcr_data.get('last_date', 'N/A')}\n")
        lines.append(f"\n*Fuente: CBOE Official Data. Timestamp: {pcr_data.get('timestamp', 'N/A')}.*\n\n")

    # --- Market Transition Engine (MTE v1.0) ---
    if mte_result:
        lines.append("## Market Transition Engine (MTE v1.0)\n")
        lines.append(f"- **Escenario:** {mte_result.get('scenario', 'N/A')} (Confianza: {mte_result.get('confidence', 0):.0%})\n")
        lines.append(f"- **Market Stress Index (MSI):** {mte_result.get('msi', 0):.0f}\n")
        lines.append(f"- **Inflation Pressure Index (IPI):** {mte_result.get('ipi', 0):.0f}\n")
        lines.append(f"- **Sector Rotation Score:** {mte_result.get('srs', 0):.2f}\n")
        lines.append(f"- **Safe Haven Score:** {mte_result.get('shs', 0):.2f}\n")
        lines.append(f"- **Credit Stress Score:** {mte_result.get('cls', 0):.2f}\n")
        lines.append(f"- **Inflation Pressure Score:** {mte_result.get('ips', 0):.2f}\n\n")

    # --- Dark Pools (FINRA ATS) ---
    if darkpool_data:
        lines.append("## Flujos Institucionales (Dark Pools v1.0)\n")
        lines.append(f"- **Dark Pool medio:** {darkpool_data.get('media_dark_pool', 0):.2f}% "
                     f"({darkpool_data.get('n_tickers_ats', 0)}/{darkpool_data.get('n_tickers_total', 0)} tickers)\n")
        if pd.notna(darkpool_data.get('z_score')):
            lines.append(f"- **Robust Z-Score:** {darkpool_data['z_score']:.2f}\n")
            lines.append(f"- **Momentum:** {darkpool_data.get('momentum', 0):.2f}\n")
            lines.append(f"- **Percentil:** {darkpool_data.get('percentile', 0):.0f}%\n")
            lines.append(f"- **Estado:** {darkpool_data.get('state', 'N/A')}\n")
        else:
            lines.append("- *Acumulando historial (se necesitan 104 semanas para el Z-Score)*\n")
        lines.append(f"- **Semana FINRA:** {darkpool_data.get('week', 'N/A')}\n")

        if 'datos' in darkpool_data and not darkpool_data['datos'].empty:
            top5 = darkpool_data['datos'].nlargest(5, 'dark_pool_pct')
            lines.append("\n**Top 5 por Dark Pool %:**\n")
            lines.append("| Ticker | Dark Pool % | Vol ATS | Vol Total |\n")
            lines.append("|--------|:-----------:|:-------:|:---------:|\n")
            for _, row in top5.iterrows():
                lines.append(f"| {row['ticker']} | {row['dark_pool_pct']:.2f}% | {row['ats_volume']:,.0f} | {row['total_volume']:,.0f} |\n")

        lines.append(f"\n*Fuente: FINRA ATS Transparency Data.*\n\n")

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
