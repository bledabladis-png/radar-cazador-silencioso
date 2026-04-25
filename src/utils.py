import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import tickers

def plot_flow_dispersion(flow_mom, output_path='outputs/flow_dispersion.png'):
    disp = flow_mom.std(axis=1).iloc[-60:]
    if len(disp) < 5:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(disp.index, disp.values, color='blue', linewidth=2)
    plt.axhline(y=disp.quantile(0.7), color='red', linestyle='--', label='Umbral alto (70%)')
    plt.axhline(y=disp.quantile(0.3), color='green', linestyle='--', label='Umbral bajo (30%)')
    plt.title('Dispersion de flujos sectoriales (ultimos 60 dias)')
    plt.xlabel('Fecha')
    plt.ylabel('Dispersion (std)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def interpret_flow_intensity(flow_z):
    if flow_z > 2.0:
        return "Fuerte entrada"
    elif flow_z > 0.8:
        return "Entrada moderada"
    elif flow_z < -2.0:
        return "Fuerte salida"
    elif flow_z < -0.8:
        return "Salida moderada"
    else:
        return "Neutral"

def save_markdown_report(ranking_price, ranking_flow, flow_dispersion, flow_breadth, regime_flow,
                         dispersion_price, breadth_price, vix_z, regime_price, accion_price,
                         alertas, output_path='outputs/reporte_diario.md'):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Radar de Rotacion Sectorial v3.15\n\n")
        f.write(f"**Fecha:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Ranking por Momentum de Precio\n\n")
        f.write("| Posicion | Sector | Momentum |\n")
        f.write("|----------|--------|----------|\n")
        for i, (sec, mom) in enumerate(ranking_price, 1):
            arrow = "↑" if mom > 0 else "↓"
            f.write(f"| {i} | {sec} | {mom:.4f} |\n")

        f.write("\n## Ranking por Flujo de Dinero Estimado\n\n")
        f.write("| Posicion | Sector | Flujo (z-score) | Interpretacion |\n")
        f.write("|----------|--------|-----------------|----------------|\n")
        for i, (sec, flow) in enumerate(ranking_flow.items(), 1):
            arrow = "↑" if flow > 0 else "↓"
            f.write(f"| {i} | {sec} | {flow:.2f} | {interpret_flow_intensity(flow)} |\n")

        f.write("\n## Alertas del Dia\n\n")
        for alerta in alertas:
            f.write(f"- {alerta}\n")

        f.write("\n## Metricas de Flujo Institucional\n\n")
        f.write(f"- **Dispersion de flujos:** {flow_dispersion:.3f}\n")
        if flow_dispersion > 0.5:
            f.write("- *Interpretacion: Alta dispersion (rotacion)*\n")
        elif flow_dispersion > 0.2:
            f.write("- *Interpretacion: Dispersion moderada (transicion)*\n")
        else:
            f.write("- *Interpretacion: Baja dispersion (trend)*\n")
        f.write(f"- **Breadth de flujo (entrada):** {flow_breadth:.1%}\n")
        f.write(f"- **Regimen de flujos:** {regime_flow}\n")

        f.write("\n## Metricas de Precio (contexto)\n\n")
        f.write(f"- **Dispersion sectorial:** {dispersion_price:.4f}\n")
        f.write(f"- **Breadth signal:** {breadth_price:.3f}\n")
        f.write(f"- **VIX z-score:** {vix_z:.2f}\n")
        if vix_z > 1.5:
            f.write("- ⚠️ **ALTA VOLATILIDAD:** Las señales pueden ser menos fiables. Interpretar con cautela.\n")
        elif vix_z > 1.0:
            f.write("- ⚡ **Volatilidad elevada:** Señales menos consistentes. Verificar con otras métricas.\n")
        f.write(f"- **Regimen de precio:** {regime_price}\n")
        f.write(f"- **Accion sugerida:** {accion_price}\n")

        f.write("\n## Visualizacion\n\n")
        f.write("![Dispersion de flujos](flow_dispersion.png)\n\n")

        f.write("\n## Conclusion\n\n")
        f.write(f"**Dinero fuerte entrando en:** {ranking_flow.index[0]}\n")
        f.write(f"**Dinero saliendo de:** {ranking_flow.index[-1]}\n")
        f.write("\n---\n*Generado automaticamente por el Radar de Rotacion Sectorial v3.15*")