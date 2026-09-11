# -*- coding: utf-8 -*-
"""Contexto de mercado: liderazgo interno, rotacion, dispersion,
correlacion y cross-asset.

Extraidos de src/report_generator.py (refactor C1, fase C1-6d3b).
"""

import pandas as pd


def render_liderazgo_interno(rs_internal_data):
    """Renderiza la tabla Liderazgo relativo interno.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if rs_internal_data is not None and not rs_internal_data.empty:
        _rs_latest = rs_internal_data[pd.to_datetime(rs_internal_data['date']) == pd.to_datetime(rs_internal_data['date']).max()]
        top_tickers = set()
        for sector in _rs_latest['sector'].unique():
            top = _rs_latest[_rs_latest['sector'] == sector].nlargest(5, 'price_ret_20d')['ticker']
            top_tickers.update(top)
        report_df = _rs_latest[_rs_latest['ticker'].isin(top_tickers)]
        out.append("## Liderazgo relativo interno\n")
        out.append("| Sector | Ticker | vs mercado 20d | vs sector 20d | Clasificación |\n")
        out.append("|--------|--------|----------------|---------------|----------------|\n")
        for _, row in report_df.iterrows():
            out.append(f"| {row['sector']} | {row['ticker']} | {row['rs_abs_20d']:.2%} | {row['rs_internal_20d']:.2%} | {row['classification']} |\n")
        out.append("\n")
    return out


def render_rotacion_reciente(sector_rank_deltas_data):
    """Renderiza la tabla Rotacion sectorial reciente.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if sector_rank_deltas_data is not None and not sector_rank_deltas_data.empty:
        out.append("## Rotación sectorial reciente\n")
        out.append("| Sector | Rank actual | Δ5d | Δ10d | Δ20d | Lectura 5d | Lectura 10d | Lectura 20d |\n")
        out.append("|--------|-------------|-----|------|------|------------|-------------|-------------|\n")
        for _, row in sector_rank_deltas_data.iterrows():
            out.append(f"| {row['sector']} | {row['rank_actual']} | {row['rank_change_5d']:+.0f} | {row['rank_change_10d']:+.0f} | {row['rank_change_20d']:+.0f} | {row['lectura_5d']} | {row['lectura_10d']} | {row['lectura_20d']} |\n")
        out.append("\n")
    return out


def render_dispersion_sectores(sector_dispersion_data):
    """Renderiza la tabla Dispersion entre sectores.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if sector_dispersion_data is not None and not sector_dispersion_data.empty:
        out.append("## Dispersión entre sectores\n")
        out.append("| Fecha | Rango (pp) | Desv (pp) | Media (pp) | Lectura | Heterogeneidad |\n")
        out.append("|-------|------------|-----------|------------|---------|----------------|\n")
        for _, row in sector_dispersion_data.iterrows():
            date_str = pd.Timestamp(row['date']).strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/D'
            out.append(f"| {date_str} | {row['range_pp']:.2f} | {row['std_pp']:.2f} | {row['mean_ret']:.2f} | {row['dispersion_reading']} | {row['heterogeneity_type']} |\n")
        out.append("\n")
        out.append("*La dispersión mide la separación entre los retornos de los 11 sectores. No es un score ni una señal.*\n\n")
    return out


def render_correlacion_sectores(sector_correlation_summary_data):
    """Renderiza la tabla Correlacion entre sectores.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if sector_correlation_summary_data is not None and not sector_correlation_summary_data.empty:
        _corr_latest = sector_correlation_summary_data[pd.to_datetime(sector_correlation_summary_data['date']) == pd.to_datetime(sector_correlation_summary_data['date']).max()]
        out.append("## Correlación entre sectores\n")
        out.append("| Ventana | Media | Mediana | P25 | P75 | Mín | Máx | Lectura |\n")
        out.append("|---------|-------|---------|-----|-----|-----|-----|---------|\n")
        for _, row in _corr_latest.iterrows():
            out.append(f"| {int(row['window'])}d | {row['corr_mean']:.2f} | {row['corr_median']:.2f} | {row['corr_p25']:.2f} | {row['corr_p75']:.2f} | {row['corr_min']:.2f} | {row['corr_max']:.2f} | {row['correlation_reading']} |\n")
        out.append("\n")
        out.append("*La correlación mide el co-movimiento entre retornos sectoriales. No es un score ni una señal.*\n\n")
    return out


def render_contexto_cross_asset(cross_asset_context_data):
    """Renderiza la tabla Contexto transversal de mercado.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if cross_asset_context_data is not None and not cross_asset_context_data.empty:
        out.append("## Contexto transversal de mercado\n")
        out.append("| Sector | Ventana | Equity | Rates | Crédito | Commodities | FX | VIX |\n")
        out.append("|--------|---------|--------|-------|---------|-------------|----|-----|\n")
        # Pivotar: para cada sector y ventana, extraer mean_corr por asset_class
        grouped = cross_asset_context_data.groupby(['sector','window','asset_class'])['mean_corr'].first().unstack()
        for (sector, window), row in grouped.iterrows():
            equity = row.get('equity', None)
            rates = row.get('rates', None)
            credit = row.get('credit', None)
            commodities = row.get('commodities', None)
            fx = row.get('fx', None)
            volatility = row.get('volatility', None)
            def _fmt(v):
                return f"{v:.2f}" if pd.notna(v) else "N/D"
            out.append(f"| {sector} | {int(window)}d | {_fmt(equity)} | {_fmt(rates)} | {_fmt(credit)} | {_fmt(commodities)} | {_fmt(fx)} | {_fmt(volatility)} |\n")
        out.append("\n")
        out.append("*Contexto descriptivo basado en correlaciones sector-activo transversal. No implica confirmación ni causalidad.*\n\n")
    return out
