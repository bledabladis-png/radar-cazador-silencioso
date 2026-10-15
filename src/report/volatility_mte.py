# -*- coding: utf-8 -*-
"""Volatilidad y Market Transition Engine (MTE v1.0).

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
            # A4-b (2026-09-28): guard NaN en los 6 campos numericos.
            # Bug observado en el reporte del 2026-09-28: la fila
            # 2026-09-14 tiene term_structure_ratio=NaN y se renderizaba
            # como 'nan' literal (f'{nan:.2f}' -> 'nan'). El resto de
            # campos numericos no tienen NaN hoy pero el defecto es
            # latente. Fix: extender el mismo patron _fmt_num ya usado
            # en otros renders del paquete.
            def _fmt(v):
                return f"{v:.2f}" if pd.notna(v) else "N/D"
            out.append(
                f"| {date_str} | {_fmt(row['vix_level'])} | {_fmt(row['vix_percentile_20d'])} | "
                f"{_fmt(row['vix_percentile_60d'])} | {_fmt(row['term_structure_ratio'])} | "
                f"{_fmt(row['pcr_zscore'])} | {_fmt(row['pcr_percentile_20d'])} | "
                f"{row['volatility_reading']} | {row['term_structure_reading']} |\n"
            )
        out.append("\n")
        out.append("*Estructura descriptiva de volatilidad implícita y posicionamiento en opciones. No incluye Dark Pool.*\n\n")
        # N6.2 (2026-10-13, auditor): ventanas exactas de cada percentil.
        # - Perc PCR (esta columna): pcr_percentile_20d, ventana 20 sesiones.
        # - Percentil (Sentimiento de Opciones): percentile, ventana
        #   min(252, len) sesiones. Distinta ventana, mismo indicador.
        out.append("*Perc PCR = percentil de PCR sobre ventana de 20 "
                   "sesiones (columna pcr_percentile_20d). El 'Percentil' "
                   "de Sentimiento de Opciones usa la misma serie con "
                   "ventana min(252, historico disponible). Miden cosas "
                   "distintas: corto plazo vs largo plazo. Pueden diferir.*\n\n")
        # H4 (2026-10-15, auditoria externa): aclarar que la 'Lectura
        # volatilidad' de esta tabla mide el NIVEL del VIX (percentil
        # del valor absoluto), mientras que el regimen de volatilidad
        # del Resumen de Regimenes mide la VELOCIDAD (z-score del
        # retorno del VIX). Son ortogonales: nivel medio con velocidad
        # baja es un estado valido. No hay contradiccion.
        out.append("*La 'Lectura volatilidad' mide el NIVEL del VIX "
                   "(percentil del valor absoluto sobre 20 sesiones). Es "
                   "distinta del regimen de volatilidad del Resumen de "
                   "Regimenes, que mide la VELOCIDAD de cambio (z-score "
                   "del retorno del VIX). Pueden coincidir o no: nivel "
                   "medio con velocidad baja es un estado valido.*\n\n")

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
        # P2.5 (2026-10-13, auditor): nota sin umbrales hardcodeados.
        # Si cambian en scoring.py, esta nota sigue siendo cierta.
        # H15 (2026-10-15, auditoria externa): publicar componentes,
        # pesos y umbrales del scoring de escenarios.
        out.append("*El escenario se asigna por suma ponderada de "
                   "condiciones. Pesos (SCENARIO_WEIGHTS): CLS=3, "
                   "SHS=2, SRS=2, IPS=3. MIXED es una constante interna "
                   "(=2). Umbrales de activacion: CLS>0.20 suma su peso; "
                   "SHS>0.20; SRS>0.10; IPS>0.15. Bonos adicionales: "
                   "CLS>0.50 suma +6 a CRISIS; CLS>0.85 suma +3; "
                   "IPS>0.50 suma +2 a STAGFLATION. STAGFLATION se "
                   "activa cuando IPS>0.15 AND SRS>0; CLS<0.30 refuerza. "
                   "La etiqueta se publica como UNCONFIRMED cuando la "
                   "confidence < 0.50. Fuente: "
                   "indicators/mte/scoring.py::score_scenarios y "
                   "SCENARIO_WEIGHTS. 'Inflation Pressure: N/5' es un "
                   "contador de componentes en presion, distinto del "
                   "score agregado Inflation Pressure Score.*\n\n")
    return out
