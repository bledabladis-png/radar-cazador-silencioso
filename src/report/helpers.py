from src.utils import resolve_reference_date
# -*- coding: utf-8 -*-
"""Helpers de formateo y clasificacion para el reporte diario.

Extraidos de src/report_generator.py (refactor C1).
"""

import pandas as pd

from config.settings import (
    EXPECTED_SECTOR_COUNT,
    FRESHNESS_DEFAULT,
    FRESHNESS_FINRA,
    FRESHNESS_FRED,
)


def _fmt_num(v, fmt="{:.2f}"):
    """Formatea un numero. NaN -> N/D.

    FU-003a-bis (2026-10-02): normaliza ruido numerico -0.00/-0.0/-0
    a 0.00/0.0/0. El signo de un cero visual es enganoso, y el ruido
    de coma flotante (p.ej. -1e-17) formateado a 2 decimales produce
    '-0.00'. Mismo contrato que _fmt_signed pero para fmt sin signo.
    """
    if pd.isna(v):
        return "N/D"
    try:
        s = fmt.format(v)
        if s.startswith('-'):
            rest = s[1:]
            if rest and all(c in '0.,%' for c in rest):
                s = rest
        return s
    except Exception:
        return str(v)


def _fmt_signed(v, fmt_signed, fmt_unsigned):
    """FU-003a (2026-09-15): cero sin signo, no-cero con signo.

    Regla: si el valor, formateado con la precision del fmt unsigned,
    resulta en todos sus digitos igual a cero, se muestra sin signo.
    Esto neutraliza:
      - El ruido numerico de coma flotante (1.1 - 1.1 != 0 exacto).
      - Los valores absolutos menores que la precision de display.
    Cuando el valor mostrado es "cero visual", un signo '+/-' delante
    seria enganoso.

    Se reciben DOS formatos explicitos. NO se manipula la cadena.
    """
    if pd.isna(v):
        return "N/D"
    try:
        formatted_unsigned = fmt_unsigned.format(abs(v))
        digits_only = ''.join(c for c in formatted_unsigned if c.isdigit())
        if not digits_only.strip('0'):
            return fmt_unsigned.format(0)
        return fmt_signed.format(v)
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
    # Politica B3 (2026-09-12): ad_net=0 con advances=declines>0 es
    # balance real -> '+0'. Distinto de 0/0 -> N/D. El '+' comunica
    # 'cero neto con datos disponibles'. NO es una violacion de FU-003a
    # (que aplica a ruido numerico y a ceros sin informacion).
    return _fmt_num(ad_net, fmt)


def _classify_freshness(age_days, max_current=None, max_recent=None, max_stale=None):
    if max_current is None:
        max_current = FRESHNESS_DEFAULT[0]
    if max_recent is None:
        max_recent = FRESHNESS_DEFAULT[1]
    if max_stale is None:
        max_stale = FRESHNESS_DEFAULT[2]
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
    max_current, max_recent, max_stale = FRESHNESS_FINRA
    if age_days <= max_current:
        return 'CURRENT'
    elif age_days <= max_recent:
        return 'RECENT'
    elif age_days <= max_stale:
        return 'STALE'
    return 'ARCHIVAL'


def _generate_coverage_table(pcr_data, darkpool_data, sector_results, reference_date=None):
    _ref = resolve_reference_date(reference_date)
    lines = []
    lines.append("### Cobertura de Datos\n")
    lines.append("| Fuente | Cobertura | Antigüedad |\n")
    lines.append("|--------|-----------|------------|\n")
    sectores_total = EXPECTED_SECTOR_COUNT
    sectores_validos = sectores_total
    if sector_results and 'ranking' in sector_results:
        sectores_validos = len([s for s in sector_results['ranking'] if s[1] is not None])
    lines.append(f"| Sectores | {sectores_validos}/{sectores_total} ({sectores_validos/sectores_total:.0%}) | - |\n")
    # D36 (2026-09-30): try/except para strings no parseables.
    # Bug: render_data_freshness protegia pcr_data['last_date'] con
    # try/except, pero esta tabla no. Con un last_date invalido
    # (Yahoo/CBOE devolviendo algo raro), el reporte entero caia
    # con DateParseError. Mismo contrato que la seccion padre.
    if pcr_data and pcr_data.get('last_date'):
        import pandas as pd
        try:
            pcr_age = (_ref - pd.Timestamp(pcr_data['last_date'])).days
            lines.append(f"| Opciones (CBOE) | - | {pcr_age} dias |\n")
        except (ValueError, TypeError):
            lines.append("| Opciones (CBOE) | - | N/D |\n")
    else:
        lines.append("| Opciones (CBOE) | - | Sin datos |\n")
    if darkpool_data and darkpool_data.get('week'):
        import pandas as pd
        try:
            dp_age = (_ref - pd.Timestamp(darkpool_data['week'])).days
            lines.append(f"| Dark Pool (FINRA) | - | {dp_age} dias |\n")
        except (ValueError, TypeError):
            lines.append("| Dark Pool (FINRA) | - | N/D |\n")
    else:
        lines.append("| Dark Pool (FINRA) | - | Sin datos |\n")
    lines.append("\n")
    return lines


def _classify_fred_freshness(age_days):
    max_current, max_recent, max_stale = FRESHNESS_FRED
    if age_days <= max_current:
        return 'CURRENT'
    elif age_days <= max_recent:
        return 'RECENT'
    elif age_days <= max_stale:
        return 'STALE'
    return 'ARCHIVAL'

def _fmt_ihr_bands():
    """Texto de bandas IHR derivado de IHR_THRESHOLDS + IHR_LABELS."""
    # H5.3 B6 (2026-10-31): antes era un literal hardcoded en sentiment.py
    # con 3 bandas que ya divergian de classify_ihr (5 categorias reales).
    from config.settings import IHR_THRESHOLDS
    from indicators.options_metrics import IHR_LABELS
    ordenados = sorted(IHR_THRESHOLDS.items(), key=lambda kv: -kv[1])
    segmentos = []
    for i, (cat, thr) in enumerate(ordenados):
        label = IHR_LABELS.get(cat, cat)
        if i == 0:
            segmentos.append(f">={thr:g} {label}")
        else:
            thr_prev = ordenados[i-1][1]
            segmentos.append(f"{thr:g}-{thr_prev:g} {label}")
    # Fallback: categoria de IHR_LABELS ausente de IHR_THRESHOLDS (else de classify_ihr).
    cats_en_thr = {cat for cat, _ in ordenados}
    cats_fallback = [c for c in IHR_LABELS if c not in cats_en_thr]
    if len(cats_fallback) == 1:
        min_thr = ordenados[-1][1]
        segmentos.append(f"<{min_thr:g} {IHR_LABELS[cats_fallback[0]]}")
    segmentos.reverse()
    return ", ".join(segmentos)
