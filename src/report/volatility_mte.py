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
            # H16 (2026-10-16, auditoria): homogeneizar unidades. Los
            # percentiles del VIX vienen en escala [0,1]; el percentil del
            # PCR viene en escala [0,100]. Se publican todos como [0,100]
            # con sufijo '%' para evitar lecturas cruzadas erroneas.
            def _pct_0_1(v):
                return f"{v*100:.0f}%" if pd.notna(v) else "N/D"
            def _pct_0_100(v):
                return f"{v:.0f}%" if pd.notna(v) else "N/D"
            out.append(
                f"| {date_str} | {_fmt(row['vix_level'])} | {_pct_0_1(row['vix_percentile_20d'])} | "
                f"{_pct_0_1(row['vix_percentile_60d'])} | {_fmt(row['term_structure_ratio'])} | "
                f"{_fmt(row['pcr_zscore'])} | {_pct_0_100(row['pcr_percentile_20d'])} | "
                f"{row['volatility_reading']} | {row['term_structure_reading']} |\n"
            )
        out.append("\n")
        out.append("*Estructura descriptiva de volatilidad implícita y posicionamiento en opciones. No incluye Dark Pool.*\n\n")
        # N6.2 (2026-10-13, auditor): ventanas exactas de cada percentil.
        # - Perc PCR (esta columna): pcr_percentile_20d, ventana 20 sesiones.
        # - Percentil (Sentimiento de Opciones): percentile, ventana
        #   min(252, len) sesiones. Distinta ventana, mismo indicador.
        # H16 (2026-10-16, auditoria): declarar unidades de cada percentil.
        out.append("*H16 auditoria 2026-10-16: Perc 20d / Perc 60d = "
                   "percentiles del VIX (escala original 0-1, publicados "
                   "aqui como 0-100%). Perc PCR = percentil del PCR sobre "
                   "20 sesiones (escala original 0-100). Unidades "
                   "homogeneizadas a % para lectura directa.*\n\n")
        out.append("*Perc PCR = percentil de PCR sobre ventana de 20 "
                   "sesiones (columna pcr_percentile_20d). El 'Percentil' "
                   "de Sentimiento de Opciones usa la misma serie con "
                   "ventana min(252, historico disponible). Miden cosas "
                   "distintas: corto plazo vs largo plazo. Pueden diferir.*\n\n")
        # H15-vol (2026-10-17, REP-AUDIT-20261017-capa6-commitE):
        # mismo bug que header.py. El regimen de volatilidad mide
        # std(returns, window=20) normalizada, NO el retorno diario del
        # VIX. Texto corregido para no publicar una definicion falsa.
        out.append("*La 'Lectura volatilidad' mide el NIVEL del VIX "
                   "(percentil del valor absoluto sobre 20 sesiones). Es "
                   "distinta del regimen de volatilidad del Resumen de "
                   "Regimenes, que mide la VOLATILIDAD REALIZADA "
                   "(desviacion estandar rolling de retornos diarios del "
                   "VIX, ventana 20, normalizada contra baseline de 3 anos). "
                   "Pueden coincidir o no: nivel medio con volatilidad "
                   "realizada baja es un estado valido.*\n\n")

    return out


def render_mte(mte_result):
    """Renderiza la seccion Market Transition Engine (MTE v1.0).

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if mte_result:
        out.append("## Market Transition Engine (MTE v1.0)\n")
        # H7 B6 (2026-10-31): V2/V3/V4. conf NaN -> UNCONFIRMED; msi/ipi/cls con guard; helpers locales.
        def _f(v, fmt):
            return format(v, fmt) if pd.notna(v) else "N/D"
        def _pct(v):
            return format(v, ".0%") if pd.notna(v) else "N/D"
        mte_conf = mte_result.get("confidence", 0)
        mte_conf_str = _f(mte_conf, ".2f")
        mte_scenario = mte_result.get("scenario", "N/A")
        _confirmed = pd.notna(mte_conf) and mte_conf >= 0.5
        if not _confirmed:
            out.append(f"- **Escenario (UNCONFIRMED):** {mte_scenario} (Confidence Score no calibrado: {mte_conf_str}) - *No se considera confirmado.*\n")
        else:
            out.append(f"- **Escenario:** {mte_scenario} (Confidence Score no calibrado: {mte_conf_str})\n")
        out.append("*Nota: Confidence Score (no calibrado, escala 0-1) representa la distancia a los umbrales y el consenso entre motores. No debe interpretarse como probabilidad.*\n")
        out.append(f"- **Market Stress Index (MSI):** {_f(mte_result.get("msi", 0), ".0f")}\n")
        out.append(f"- **Inflation Pressure Index (IPI):** {_f(mte_result.get("ipi", 0), ".0f")}\n")
        out.append(f"- **Sector Rotation Score:** {_f(mte_result.get("srs", 0), ".2f")}\n")
        out.append(f"- **Safe Haven Score:** {_f(mte_result.get("shs", 0), ".2f")}\n")
        out.append(f"- **Credit Stress Score:** {_f(mte_result.get("cls", 0), ".2f")} (orientacion: positivo = mayor estres crediticio)\n")
        out.append(f"- **Inflation Pressure Score:** {_f(mte_result.get("ips", 0), ".2f")}\n")
        # Fix 2026-11-04: exponer mte_state si no es VALID (hallazgo 10.2).
        # El state interno persiste VALID/DEGRADED/INVALID. El render
        # publica el estado cuando no es VALID para que el lector sepa
        # que el escenario se calculo con inputs degradados.
        _mte_state = mte_result.get('mte_state')
        if _mte_state and _mte_state != 'VALID':
            out.append(
                f"- **MTE Status:** {_mte_state} "
                "*(escenario calculado con inputs degradados o insuficientes; "
                "ver Futures Status y cobertura para detalle)*\n")
        # Fix 2026-10-30: exponer futures_status si no es OK. El state
        # interno (outputs/state/mte_state.json) declara STALE/UNAVAILABLE
        # cuando los precios de futuros no estan frescos, pero el reporte
        # no lo publicaba. El escenario MTE usa futuros como input; el
        # lector debe saber si estan frescos o no. La confidence no se
        # degrada por este estado (fix queda fuera del scoring).
        _fut_status = mte_result.get('futures_status')
        if _fut_status and _fut_status != 'OK':
            out.append(
                f"- **Futures Status:** {_fut_status} "
                "*(inputs de futuros no frescos o ausentes; "
                "la confidence del escenario NO se degrada por este estado)*\n")
        # Fix 2026-10-30 (clase 1fb791ab): pending_out es el escenario
        # al que se esta transitando. Si no es None, el scenario actual
        # es el estado consolidado y pending es la transicion en curso
        # que se confirmara en el siguiente run si persisten las
        # condiciones. Sin esta linea, un lector ve dos dias con los
        # mismos indices (msi, ipi, srs, shs, cls, ips) y distinto
        # scenario sin saber por que (ejemplo: run 41037457375 tenia
        # pending=RECESSION con scenario=SOFT LANDING; run local
        # siguiente confirmo RECESSION).
        _pending = mte_result.get('pending')
        if _pending:
            out.append(
                f"- **Pending transition:** {_pending} "
                "*(condiciones apuntan a este escenario; se confirmara "
                "en el siguiente run si persisten. El scenario actual "
                "refleja el estado consolidado)*\n")
        out.append("\n")
        # REP-AUDIT-20261018-IPS (2026-10-18): aclarar que IPS/IPI son
        # proxy de mercado, no tasa de inflacion observada (IPC/PPI).
        out.append("*Inflation Pressure Score / Index: proxy de mercado "
                   "de presiones inflacionarias, no una medida de la tasa "
                   "de inflacion observada (IPC/PPI). Inputs: momentum 20d "
                   "de XLE (energia), ^SPGSCI (commodities) y TIP (TIPS), "
                   "y ratio TIP/IEF (proxy de breakeven), normalizados por "
                   "z-score robusto 60d y comprimidos con tanh; media "
                   "EWMA(20) en [-1,+1]. IPI = (IPS+1)/2*100, mapeo lineal "
                   "a [0,100]. Puede divergir de IPC/PPI publicados sin ser "
                   "error: mide presion via precios de activos, no niveles "
                   "macro observados.*\n\n")
        out.append("*Nota cruzada: puede discrepar de la etiqueta Macro 'INFLATION SHOCK' del Resumen de Regimenes. Motores independientes (macro_regime vs MTE scoring): miden cosas distintas; signos opuestos no son error.*\n\n")
        # H7 B6 (2026-10-31): nota derivada de SCENARIO_RULES.
        from src.report.helpers import _fmt_scenario_rules_note
        out.append(_fmt_scenario_rules_note())
    return out
