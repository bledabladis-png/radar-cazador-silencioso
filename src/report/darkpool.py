# -*- coding: utf-8 -*-
"""Dark Pools (FINRA ATS Transparency Data).

Extraido de src/report_generator.py (refactor C1, fase C1-6d5).
"""

from datetime import datetime

import pandas as pd

from src.report.helpers import _classify_finra_freshness


def render_darkpool(darkpool_data):
    """Renderiza la seccion Actividad en ATS - Dark Pools (FINRA).

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if darkpool_data:
        out.append("## Actividad en ATS - Dark Pools (FINRA v1.0)\n")
        out.append("*Nota: FINRA publica datos de ATS con retraso regulatorio de 2 a 4 semanas. Los datos pueden estar desfasados por diseño.*\n")
        week = darkpool_data.get('week', 'N/A')
        if week != 'N/A':
            try:
                d = pd.Timestamp(week)
                age = (datetime.now() - d).days
                freshness = _classify_finra_freshness(age)
                if freshness == 'ARCHIVAL':
                    out.append(f"**DATOS OBSOLETOS:** Ultimo dato con {age} dias de antiguedad. No se usa para clasificacion actual. Contexto historico solamente.\n\n")
            except:
                pass
        out.append(f"- **% Volumen en ATS medio:** {darkpool_data.get('media_dark_pool', 0):.2f}% "
                     f"({darkpool_data.get('n_tickers_ats', 0)}/{darkpool_data.get('n_tickers_total', 0)} tickers)\n")
        
        z_windows = darkpool_data.get('z_windows', {})
        if z_windows:
            out.append("- **Z-Scores por ventana:**\n")
            for w_name, w_data in z_windows.items():
                if w_data:
                    out.append(f"  - {w_name}: Z={w_data['z']:.2f}, Estado={w_data['state']}\n")
        elif pd.notna(darkpool_data.get('z_score')):
            out.append(f"- **Robust Z-Score:** {darkpool_data['z_score']:.2f}\n")
            out.append(f"- **Momentum:** {darkpool_data.get('momentum', 0):.2f}\n")
            out.append(f"- **Percentil:** {darkpool_data.get('percentile', 0):.0f}%\n")
            out.append(f"- **Estado ATS:** {darkpool_data.get('state', 'N/A')}\n")
        else:
            out.append("- *Acumulando historial (se necesitan {DARKPOOL_FULL_HISTORY_WEEKS} semanas para el Z-Score)*\n")
        if week != 'N/A':
            try:
                d = pd.Timestamp(week)
                age = (datetime.now() - d).days
                out.append(f"- **Semana FINRA:** {week} (retraso: {age} dias)\n")
            except Exception:
                out.append(f"- **Semana FINRA:** {week}\n")
        else:
            out.append("- **Semana FINRA:** N/D\n")

        if 'datos' in darkpool_data and not darkpool_data['datos'].empty:
            out.append("\n**Mayor % de volumen en ATS:**\n")
            out.append("| Ticker | % ATS | Vol ATS | Vol Total |\n")
            out.append("|--------|:-----:|:-------:|:---------:|\n")
            top5 = darkpool_data['datos'].nlargest(5, 'dark_pool_pct')
            for _, row in top5.iterrows():
                out.append(f"| {row['ticker']} | {row['dark_pool_pct']:.2f}% | {row['ats_volume']:,.0f} | {row['total_volume']:,.0f} |\n")
            out.append("\n*Nota: Un alto % de volumen en ATS NO implica acumulación institucional. Las categorias reflejan el nivel de actividad ATS relativa a su historial, no la direccion del flujo institucional.*\n")
        out.append("\n*Fuente: FINRA ATS Transparency Data.*\n\n")
    return out
