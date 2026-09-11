# -*- coding: utf-8 -*-
"""Secciones sectoriales: Breadth, Concentracion, Dispersion.

Extraidas de src/report_generator.py (refactor C1, fase C1-6b).
"""

import pandas as pd

from config.settings import MIN_SECTOR_COVERAGE
from src.report.helpers import _fmt_num


def render_sector_breadth(sector_breadth_data):
    """Renderiza la tabla Sector Breadth & Health.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if sector_breadth_data is not None and not sector_breadth_data.empty:
        latest_date = pd.to_datetime(sector_breadth_data['date']).max()
        breadth_latest = sector_breadth_data[pd.to_datetime(sector_breadth_data['date']) == latest_date]
        out.append("## Sector Breadth & Health\n")
        out.append("| Sector | EMA20 | EMA50 | EMA200 | RS+ | Mom+ | Acc | Markup | Dist | Markdown | NH | NL | A/D | Cobertura |\n")
        out.append("|--------|-------|-------|--------|-----|------|-----|--------|------|----------|----|----|-----|-----------|\n")
        for _, row in breadth_latest.iterrows():
            cobertura = (row['n_valid_ema200'] / row['n_total'] * 100) if row['n_total'] else 0
            cov_label = f"{cobertura:.0f}%"
            if (cobertura / 100) < MIN_SECTOR_COVERAGE:
                cov_label += " [BAJA]"
            out.append(f"| {row['sector']} | {_fmt_num(row['pct_above_ema20'], '{:.1f}%')} | {_fmt_num(row['pct_above_ema50'], '{:.1f}%')} | {_fmt_num(row['pct_above_ema200'], '{:.1f}%')} | {_fmt_num(row['pct_rs_positive'], '{:.1f}%')} | {_fmt_num(row['pct_momentum_positive'], '{:.1f}%')} | {_fmt_num(row['count_accumulation'], '{:.0f}')} | {_fmt_num(row['count_markup'], '{:.0f}')} | {_fmt_num(row['count_distribution'], '{:.0f}')} | {_fmt_num(row['count_markdown'], '{:.0f}')} | {_fmt_num(row['new_highs'], '{:.0f}')} | {_fmt_num(row['new_lows'], '{:.0f}')} | {_fmt_num(row['ad_net'], '{:+d}')} | {cov_label} |\n")
        out.append("\n")
        out.append(f"*[BAJA] = cobertura < {MIN_SECTOR_COVERAGE:.0%} del universo del sector. Los ratios se calculan sobre la parte valida.*\n\n")
    return out


def render_sector_concentration(sector_concentration_data):
    """Renderiza la tabla de Concentracion del liderazgo.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if sector_concentration_data is not None and not sector_concentration_data.empty:
        latest_date = pd.to_datetime(sector_concentration_data['date']).max()
        conc_latest = sector_concentration_data[pd.to_datetime(sector_concentration_data['date']) == latest_date]
        out.append("## Concentración del liderazgo\n")
        out.append("| Sector | Top1 | Top3 | Top5 | RS med | Mom med | Flow med | Wyckoff med | WLS med | Líder | Ret Líder | Cob RS | Cob Mom | Cob Flow | Cob Wyckoff | Cob WLS |\n")
        out.append("|--------|------|------|------|--------|---------|----------|-------------|---------|-------|----------|--------|---------|----------|-------------|---------|\n")
        for _, row in conc_latest.iterrows():
            out.append(f"| {row['sector']} | {_fmt_num(row['top1_positive_return_concentration'], '{:.1%}')} | {_fmt_num(row['top3_positive_return_concentration'], '{:.1%}')} | {_fmt_num(row['top5_positive_return_concentration'], '{:.1%}')} | {_fmt_num(row['rs_median'], '{:.4f}')} | {_fmt_num(row['momentum_median'], '{:.2%}')} | {_fmt_num(row['flow_median'], '{:.2f}')} | {_fmt_num(row['wyckoff_median'], '{:.2f}')} | {_fmt_num(row['wls_median'], '{:.2f}')} | {row['leader_ticker']} | {_fmt_num(row['leader_return20'], '{:.2%}')} | {_fmt_num(row['coverage_rs'], '{:.0f}%')} | {_fmt_num(row['coverage_momentum'], '{:.0f}%')} | {_fmt_num(row['coverage_flow'], '{:.0f}%')} | {_fmt_num(row['coverage_wyckoff'], '{:.0f}%')} | {_fmt_num(row['coverage_wls'], '{:.0f}%')} |\n")
        out.append("\n")
    return out


def render_sector_dispersion(sector_concentration_data):
    """Renderiza la tabla de Dispersion interna.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if sector_concentration_data is not None and not sector_concentration_data.empty:
        latest_date = pd.to_datetime(sector_concentration_data['date']).max()
        disp_latest = sector_concentration_data[pd.to_datetime(sector_concentration_data['date']) == latest_date]
        out.append("## Dispersión interna\n")
        out.append("| Sector | RS P25 | RS Med | RS P75 | Mom P25 | Mom Med | Mom P75 |\n")
        out.append("|--------|--------|--------|--------|---------|---------|---------|\n")
        for _, row in disp_latest.iterrows():
            out.append(f"| {row['sector']} | {row['rs_p25']:.4f} | {row['rs_median']:.4f} | {row['rs_p75']:.4f} | {row['momentum_p25']:.2%} | {row['momentum_median']:.2%} | {row['momentum_p75']:.2%} |\n")
        out.append("\n")
        out.append("*Los percentiles de Flow y WLS están disponibles en outputs/history/sector_concentration.csv.*\n\n")
    return out
