# -*- coding: utf-8 -*-
"""Sentimiento de Opciones (CBOE PCR).

Extraido de src/report_generator.py (refactor C1, fase C1-6d2).
"""

from datetime import datetime

import numpy as np
import pandas as pd

from src.report.helpers import _fmt_num


def render_sentimiento_opciones(pcr_data):
    """Renderiza la seccion Sentimiento de Opciones.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if pcr_data:
        out.append("## Sentimiento de Opciones\n")
        out.append(f"- **PCR Total:** {pcr_data.get('total_pcr', np.nan):.2f} ")
        ewma_val = pcr_data.get('pcr_ewm', np.nan)
        if pd.notna(ewma_val):
            out.append(f"(EWMA(5): {ewma_val:.2f})\n")
        else:
            out.append("(EWMA(5): N/D - historial insuficiente)\n")
        if pd.notna(pcr_data.get('z_score')):
            out.append(f"- **Robust Z-Score:** {pcr_data['z_score']:.2f}\n")
            out.append(f"- **Momentum:** {pcr_data.get('momentum', 0):.2f}\n")
            out.append(f"- **Percentil:** {pcr_data.get('percentile', 0):.0f}%\n")
            out.append(f"- **Estado:** {pcr_data.get('state', 'N/A')}\n")
        out.append(f"- **PCR Indices:** {_fmt_num(pcr_data.get('index_pcr', np.nan), '{:.2f}')} | "
                     f"**PCR Acciones:** {pcr_data.get('equity_pcr', np.nan):.2f} | "
                     f"**PCR ETP:** {pcr_data.get('etp_pcr', np.nan):.2f}\n")
        out.append(f"- **PCR VIX:** {pcr_data.get('vix_pcr', np.nan):.2f} | "
                     f"**PCR SPX:** {pcr_data.get('spx_pcr', np.nan):.2f}\n")
        out.append(f"- **Institutional Hedge Ratio:** {pcr_data.get('ihr', np.nan):.2f} "
                     f"({pcr_data.get('ihr_state', 'N/A')}, bandas: <1.2 Especulacion, 1.2-1.6 Equilibrado, >1.6 Cobertura institucional)\n")
        out.append(f"- **Volumen en Indices:** {pcr_data.get('index_volume_share', np.nan):.1%} del total\n")
        out.append(f"- **Put Share:** {pcr_data.get('put_share', np.nan):.1%} | "
                     f"**Call Share:** {pcr_data.get('call_share', np.nan):.1%}\n")
        out.append(f"- **Volume PCR (calculado):** {pcr_data.get('volume_pcr', np.nan):.2f} | "
                     f"**OI PCR:** {pcr_data.get('oi_pcr', np.nan):.2f}\n")
        last_date = pcr_data.get('last_date', 'N/A')
        out.append(f"- **Ultimo dato:** {last_date}")
        if last_date != 'N/A':
            try:
                data_date = pd.Timestamp(last_date)
                age = (datetime.now() - data_date).days
                out.append(f" (desfase: {age} dias)")
            except:
                pass
        out.append("\n")
        out.append(f"\n*Fuente: CBOE Official Data. Timestamp: {pcr_data.get('timestamp', 'N/A')}.*\n\n")
    return out
