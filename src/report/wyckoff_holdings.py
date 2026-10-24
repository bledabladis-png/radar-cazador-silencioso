"""Renderizado de las secciones 'Tickers en acumulacion/distribucion'.

Consume el dict de tablas producido por
src.pipeline.wyckoff_holdings.compute_wyckoff_holdings (o por
indicators.wyckoff_holdings_report.compute_wyckoff_holdings_report)
y produce las secciones del reporte con las 3 sub-tablas por seccion.

Contrato: docs/auditoria/wyckoff/63_plan_sistema_holdings_2026-10-24.md.
"""
from __future__ import annotations

import pandas as pd


ACC_KEYS = ("usa_sector_acc", "usa_index_acc", "eu_index_acc")
DIST_KEYS = ("usa_sector_dist", "usa_index_dist", "eu_index_dist")


def render_wyckoff_holdings_acc(tables):
    """Seccion 'Tickers en acumulacion'."""
    return _render_section(
        tables,
        title="Tickers en acumulacion",
        keys=ACC_KEYS,
        group_label=("Sector", "Indice", "Indice"),
        section_note=(
            "*Acumulacion = ACCUMULATION (confirmada), ACCUMULATION_FORMING "
            "(senal temprana) o MARKUP (acumulacion consumada, precio sobre "
            "resistencia). Detalle completo en outputs/history/.*\n\n"
        ),
    )


def render_wyckoff_holdings_dist(tables):
    """Seccion 'Tickers en distribucion'."""
    return _render_section(
        tables,
        title="Tickers en distribucion",
        keys=DIST_KEYS,
        group_label=("Sector", "Indice", "Indice"),
        section_note=(
            "*Distribucion = DISTRIBUTION (confirmada), DISTRIBUTION_FORMING "
            "(senal temprana) o MARKDOWN (distribucion consumada, precio bajo "
            "soporte). Detalle completo en outputs/history/.*\n\n"
        ),
    )


def _render_section(tables, title, keys, group_label, section_note):
    if not tables:
        return []
    has_any = any(_has_data(tables, k) for k in keys)
    if not has_any:
        return []

    out = []
    out.append("\n## {}\n\n".format(title))
    subtitles = ("USA sectores", "USA indices", "Indices europeos")
    for key, subtitle, gl in zip(keys, subtitles, group_label):
        _render_subsection(out, subtitle, tables.get(key), gl)
    out.append(section_note)
    return out


def _has_data(tables, key):
    df = tables.get(key) if tables else None
    return df is not None and not df.empty


def _render_subsection(out, subtitle, df, group_label):
    out.append("### {}\n\n".format(subtitle))
    if df is None or df.empty:
        out.append("*Sin tickers en este estado.*\n\n")
        return
    n_by_phase = df.groupby("phase").size().to_dict()
    summary = ", ".join("{} {}".format(n, ph) for ph, n in n_by_phase.items())
    out.append("*{} tickers: {}.*\n\n".format(len(df), summary))
    out.append("| {} | Ticker | Fase | Desde | Peso % | Cob % | Close |\n"
               .format(group_label))
    out.append("|---|---|---|---|---|---|---|\n")
    for _, row in df.iterrows():
        out.append(
            "| {} | {} | {} | {} | {} | {} | {} |\n".format(
                row.get("group", "-"),
                row.get("ticker", "-"),
                row.get("phase", "-"),
                _fmt_since(row.get("phase_since")),
                _fmt_num(row.get("weight"), 2),
                _fmt_num(row.get("coverage_weight_pct"), 1),
                _fmt_num(row.get("close"), 2),
            )
        )
    out.append("\n")


def _fmt_since(v):
    if v is None or v == "":
        return "-"
    if isinstance(v, float) and pd.isna(v):
        return "-"
    try:
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return str(v)


def _fmt_num(v, decimals):
    if v is None:
        return "-"
    try:
        if pd.isna(v):
            return "-"
        return "{:.{}f}".format(float(v), decimals)
    except (ValueError, TypeError):
        return str(v)