# -*- coding: utf-8 -*-
"""Volatilidad, Calidad de datos y Market Transition Engine (MTE v1.0).

Extraidos de src/report_generator.py (refactor C1, fase C1-6d4c1).
"""

import pandas as pd


def render_estructura_volatilidad(volatility_structure_data):
    """Renderiza la tabla Estructura de volatilidad.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if volatility_structure_data is not None and not volatility_structure_data.empty:
        out.append("## Estructura de volatilidad\n")
        out.append("| Fecha | VIX | Perc 20d | Perc 60d | VIX3M/VIX | PCR z | Perc PCR | Lectura volatilidad | Term structure |\n")
        out.append("|-------|-----|----------|----------|-----------|-------|----------|---------------------|----------------|\n")
        for _, row in volatility_structure_data.iterrows():
            date_str = pd.Timestamp(row['date']).strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/D'
            out.append(f"| {date_str} | {row['vix_level']:.2f} | {row['vix_percentile_20d']:.2f} | {row['vix_percentile_60d']:.2f} | {row['term_structure_ratio']:.2f} | {row['pcr_zscore']:.2f} | {row['pcr_percentile_20d']:.2f} | {row['volatility_reading']} | {row['term_structure_reading']} |\n")
        out.append("\n")
        out.append("*Estructura descriptiva de volatilidad implícita y posicionamiento en opciones. No incluye Dark Pool.*\n\n")

    return out


def render_calidad_datos(data_quality_data):
    """Renderiza la tabla Calidad, frescura y cobertura de datos.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if data_quality_data is not None and not data_quality_data.empty:
        out.append("## Calidad, frescura y cobertura de datos\n")
        out.append("| Fuente | Último dato | Edad (días) | Frecuencia | Frescura | Cobertura | Notas |\n")
        out.append("|--------|-------------|--------------|------------|----------|-----------|-------|\n")
        # Mostrar solo la fila más reciente por fuente para evitar duplicados históricos
        dq = data_quality_data.copy()
        if 'date' in dq.columns and 'source' in dq.columns:
            dq['date'] = pd.to_datetime(dq['date'], errors='coerce')
            latest_idx = dq.groupby('source')['date'].idxmax()
            dq = dq.loc[latest_idx].sort_values('source')
        for _, row in dq.iterrows():
            last = row['last_date'] if pd.notna(row['last_date']) else 'N/D'
            age = f"{row['age_calendar_days']:.0f}" if pd.notna(row['age_calendar_days']) else 'N/D'
            freq = row['frequency'] if pd.notna(row['frequency']) else 'N/D'
            fresh = row['freshness'] if pd.notna(row['freshness']) else 'N/D'
            cov = f"{row['coverage']:.2%}" if pd.notna(row['coverage']) else 'N/D'
            notes = row['notes'] if pd.notna(row['notes']) else ''
            out.append(f"| {row['source']} | {last} | {age} | {freq} | {fresh} | {cov} | {notes} |\n")
        out.append("\n")
        out.append("*No todas las variables tienen la misma actualidad. Los datos se muestran sin interpolación.*\n\n")
    return out


def render_mte(mte_result):
    """Renderiza la seccion Market Transition Engine (MTE v1.0).

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if mte_result:
        out.append("## Market Transition Engine (MTE v1.0)\n")
        mte_conf = mte_result.get('confidence', 0)
        mte_conf_str = f'{mte_conf:.2f}' if pd.notna(mte_conf) else 'N/D'
        mte_scenario = mte_result.get('scenario', 'N/A')
        if mte_conf < 0.5:
            out.append(f"- **Escenario (UNCONFIRMED):** {mte_scenario} (Confidence Score no calibrado: {mte_conf_str}) - *No se considera confirmado.*\n")
        else:
            out.append(f"- **Escenario:** {mte_scenario} (Confidence Score no calibrado: {mte_conf_str})\n")
        out.append("*Nota: Confidence Score (no calibrado, escala 0-1) representa la distancia a los umbrales y el consenso entre motores. No debe interpretarse como probabilidad.*\n")
        out.append(f"- **Market Stress Index (MSI):** {mte_result.get('msi', 0):.0f}\n")
        out.append(f"- **Inflation Pressure Index (IPI):** {mte_result.get('ipi', 0):.0f}\n")
        val_srs = mte_result.get('srs', 0)
        val_srs_str = f'{val_srs:.2f}' if pd.notna(val_srs) else 'N/D'
        out.append(f"- **Sector Rotation Score:** {val_srs_str}\n")
        val_shs = mte_result.get('shs', 0)
        val_shs_str = f'{val_shs:.2f}' if pd.notna(val_shs) else 'N/D'
        out.append(f"- **Safe Haven Score:** {val_shs_str}\n")
        out.append(f"- **Credit Stress Score:** {mte_result.get('cls', 0):.2f}")
        out.append(" (orientacion: positivo = mayor estres crediticio)\n")
        ips_val = mte_result.get('ips', 0)
        ips_str = f'{ips_val:.2f}' if pd.notna(ips_val) else 'N/D'
        out.append(f"- **Inflation Pressure Score:** {ips_str}\n\n")
    return out
