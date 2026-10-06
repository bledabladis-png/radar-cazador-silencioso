# -*- coding: utf-8 -*-
"""Dark Pools (FINRA ATS Transparency Data).

Rediseno K-DT3-AUDIT-01 (2026-10-06):
- Se elimina la rama elif muerta (z_windows siempre estaba poblado,
  la rama de resumen no se ejecutaba nunca).
- `_fmt_num` en todas las lineas numericas. Antes z_score y percentile
  usaban f-string directo -> NaN salia como "nan" y "nan%".
- Se muestran SIEMPRE resumen + ventanas (antes era excluyente por bug).
- Caso STALE ademas de ARCHIVAL para datos FINRA con retraso.
- `momentum` eliminado (era copia exacta de z_score).
- `n_tickers_ats`/`n_tickers_total` -> `n_tickers` (siempre iguales).
"""
from datetime import datetime

import pandas as pd

from src.report.helpers import _classify_finra_freshness, _fmt_num
from config.settings import DARKPOOL_FULL_HISTORY_WEEKS


def render_darkpool(darkpool_data, reference_date=None):
    """Renderiza la seccion Actividad en ATS - Dark Pools (FINRA).

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if not darkpool_data:
        return out

    _ref = reference_date.replace(tzinfo=None) if reference_date is not None else datetime.now()

    out.append("## Actividad en ATS - Dark Pools (FINRA v1.0)\n")
    out.append("*Nota: FINRA publica datos de ATS con retraso regulatorio de 2 a 4 semanas. Los datos pueden estar desfasados por diseno.*\n")

    week = darkpool_data.get('week', 'N/A')
    if week != 'N/A':
        try:
            d = pd.Timestamp(week)
            age = (_ref - d).days
            freshness = _classify_finra_freshness(age)
            if freshness == 'ARCHIVAL':
                out.append(f"**DATOS OBSOLETOS:** Ultimo dato con {age} dias de antiguedad. No se usa para clasificacion actual. Contexto historico solamente.\n\n")
            elif freshness == 'STALE':
                out.append(f"**DATOS RETRASADOS:** Ultimo dato con {age} dias de antiguedad.\n\n")
        except (ValueError, TypeError):
            pass

    # Resumen: media, z oficial, percentil, estado, numero de tickers.
    out.append(f"- **% Volumen en ATS medio:** {_fmt_num(darkpool_data.get('media_dark_pool', 0), '{:.2f}%')} "
               f"({darkpool_data.get('n_tickers', 0)} tickers)\n")

    z_official = darkpool_data.get('z_score')
    if z_official is not None and pd.notna(z_official):
        out.append(f"- **Robust Z-Score:** {_fmt_num(z_official, '{:.2f}')}\n")
        out.append(f"- **Percentil:** {_fmt_num(darkpool_data.get('percentile', 0), '{:.0f}')}%\n")
        out.append(f"- **Estado ATS:** {darkpool_data.get('state', 'N/A')}\n")
    else:
        out.append(f"- *Acumulando historial (se necesitan {DARKPOOL_FULL_HISTORY_WEEKS} semanas para el Z-Score)*\n")

    # Detalle por ventana (contexto, sin `momentum`).
    z_windows = darkpool_data.get('z_windows', {})
    active_windows = {k: v for k, v in z_windows.items() if v}
    if active_windows:
        out.append("- **Z-Scores por ventana:**\n")
        for w_name, w_data in active_windows.items():
            z_val = _fmt_num(w_data.get('z'), '{:.2f}')
            out.append(f"  - {w_name}: Z={z_val}, Estado={w_data.get('state', 'N/A')}\n")

    if week != 'N/A':
        try:
            d = pd.Timestamp(week)
            age = (_ref - d).days
            out.append(f"- **Semana FINRA:** {week} (retraso: {age} dias)\n")
        except (ValueError, TypeError, OSError):
            out.append(f"- **Semana FINRA:** {week}\n")
    else:
        out.append("- **Semana FINRA:** N/D\n")

    if 'datos' in darkpool_data and not darkpool_data['datos'].empty:
        out.append("\n**Mayor % de volumen en ATS:**\n")
        out.append("| Ticker | % ATS | Vol ATS | Vol Total |\n")
        out.append("|--------|:-----:|:-------:|:---------:|\n")
        top5 = darkpool_data['datos'].nlargest(5, 'dark_pool_pct')
        for _, row in top5.iterrows():
            out.append(f"| {row['ticker']} | {_fmt_num(row['dark_pool_pct'], '{:.2f}%')} | {row['ats_volume']:,.0f} | {row['total_volume']:,.0f} |\n")
        out.append("\n*Nota: Un alto % de volumen en ATS NO implica acumulacion institucional. Las categorias reflejan el nivel de actividad ATS relativa a su historial, no la direccion del flujo institucional.*\n")

    out.append("\n*Fuente: FINRA ATS Transparency Data.*\n\n")
    return out
