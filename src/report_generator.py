import pandas as pd
import os
from datetime import datetime
from config.tickers import SECTOR_NAMES

MODEL_VERSION = "3.1"
WEIGHTS_VERSION = "3"
INDICATORS_VERSION = "2"

def generate_daily_report(macro_score, macro_regime, macro_conf, liquidity_score, liquidity_regime, liq_conf,
                          volatility_score, vol_regime, vol_conf, sector_results,
                          price_ranking, flow_ranking,
                          leader_lines=None, breadth_values=None, real_liquidity_regime=None, real_liquidity_conf=None,
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
    for i, (ticker, mom) in enumerate(otros_price[:15], 1):
        lines.append(f"| {i} | {ticker} | {mom*100:.2f}% |\n")

    lines.append("\n## Flujo Institucional - Otros Activos (Proxy)\n")
    lines.append("| # | Activo | Flujo (z-score) |\n")
    lines.append("|---|--------|------------------|\n")
    for i, (ticker, flow) in enumerate(otros_flow[:15], 1):
        lines.append(f"| {i} | {ticker} | {flow:.2f} |\n")

    if leader_lines:
        lines.append("\n## Lideres Sectoriales\n")
        lines.append("> Acciones con mejor perfil institucional dentro de sectores favorables.\n\n")
        lines.extend(leader_lines)
    else:
        lines.append("\n## Lideres Sectoriales\n")
        lines.append("*No disponibles: ningun sector en fase de acumulacion.*\n")

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
