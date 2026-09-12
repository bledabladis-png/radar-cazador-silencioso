# -*- coding: utf-8 -*-
"""Helpers de formateo y clasificacion para el reporte diario.

Extraidos de src/report_generator.py (refactor C1).
"""

import pandas as pd


def _fmt_num(v, fmt="{:.2f}"):
    if pd.isna(v):
        return "N/D"
    try:
        return fmt.format(v)
    except Exception:
        return str(v)


def _fmt_ad_net(advances, declines, ad_net, fmt="{:+d}"):
    """Formatea A/D Net con politica B3 (2026-09-12).

    Reglas:
        advances/declines invalidos (NaN/None) -> N/D.
        advances + declines == 0 -> N/D (sin informacion direccional).
        advances + declines > 0 -> formatear ad_net normalmente.

    Diferencia con ad_net == 0:
        ad_net=0 con advances=declines>0 es un balance real (mercado plano).
        ad_net=0 con advances=declines=0 es ausencia de informacion.
    """
    if pd.isna(advances) or pd.isna(declines):
        return "N/D"
    try:
        if (advances + declines) == 0:
            return "N/D"
    except Exception:
        return "N/D"
    return _fmt_num(ad_net, fmt)


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
    # Fix C22b: fallback 0 (no 110) cuando no hay CSV. El numero real
    # viene del CSV que run.py regenera solo si hay sectores favorables.
    n_acciones = 0
    try:
        import pandas as pd
        df = pd.read_csv('outputs/report/analisis_lideres.csv')
        if 'ticker' in df.columns:
            n_acciones = len(df['ticker'].unique())
    except FileNotFoundError:
        # FU-005 (2026-09-13): sin sectores favorables no se genera el CSV.
        # Es estado esperado, no un WARN.
        pass
    except Exception as e:
        print(f"  [WARN] report_generator: analisis_lideres.csv: {e}")
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


def _classify_fred_freshness(age_days):
    if age_days <= 30:
        return 'CURRENT'
    elif age_days <= 60:
        return 'RECENT'
    elif age_days <= 90:
        return 'STALE'
    return 'ARCHIVAL'
