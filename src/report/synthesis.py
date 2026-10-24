# -*- coding: utf-8 -*-
"""Índices (USA + Europa), Sintesis de Senales y Matriz de Evidencia.

Extraidos de src/report_generator.py (refactor C1, fase C1-6d6).
"""

import pandas as pd

from config.index_tickers import INDEX_CONFIG
from src.report.helpers import _fmt_num
from config.tickers import SECTOR_NAMES
from config.settings import FLOW_ZSCORE_WINDOW, EXPECTED_SECTOR_COUNT


def render_indices_internacionales(index_phases, index_leaders, index_struct_scores=None):
    """Renderiza las tablas de Índices (USA + Europa) (Wyckoff + Oportunidades).

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    out.append("\n## Índices (USA + Europa) — Fases Wyckoff\n")
    # struct_score: magnitud ademas de la etiqueta binaria. Una RANGE
    # con struct cerca de 0.30 esta proxima a MARKUP. El lector ve
    # el margen sin tener que consultar el CSV.
    _struct_map = index_struct_scores or {}
    out.append("| Indice | Ticker | Fase Wyckoff | Struct |\n")
    out.append("|--------|--------|--------------|--------|\n")
    # H5 (2026-10-21): declarar que mide struct_score para evitar la
    # confusion "MARKUP con retorno reciente negativo".
    out.append("*Nota: Struct (struct_score) mide tendencia de fondo "
               "(MA lenta + compresion ATR), no retorno reciente. Un "
               "indice en fase MARKUP puede tener retorno de corto plazo "
               "negativo; la fase y el Struct miden tendencia estructural, "
               "no momentum inmediato.*\n\n")
    if index_phases:
        for nombre, fase in index_phases.items():
            ticker = INDEX_CONFIG.get(nombre, {}).get('index_ticker', '')
            _s = _struct_map.get(nombre)
            _s_str = _fmt_num(_s, '{:+.2f}') if _s is not None else 'N/D'
            out.append(f"| {nombre} | {ticker} | {fase} | {_s_str} |\n")
    else:
        out.append("| No disponible | No disponible | No disponible | N/D |\n")
    out.append("\n")

    out.append("\n## Índices (USA + Europa) — Oportunidades de Acumulación y Markup\n")
    out.append("*Nota: Los componentes se obtienen de ETFs proxy que replican el indice de referencia. USA: SPY (S&P 500), DIA (Dow Jones), QQQ (Nasdaq-100), IWM (Russell 2000). Europa: FEZ (Euro Stoxx 50), LYXI (Ibex 35), DAXEX (DAX 40), ISF.L (FTSE 100). Solo se muestran indices en fase ACCUMULATION o MARKUP.*\n\n")
    out.append(f"*Nota: Flujo (z) es un z-score robusto sobre {FLOW_ZSCORE_WINDOW} dias. Valores extremos pueden deberse a eventos corporativos o volúmenes inusuales. Para el WLS se usa una versión normalizada limitada a ±3.*\n\n")
    if index_leaders:
        for nombre, top5 in index_leaders.items():
            if top5 is None or top5.empty:
                continue
            out.append(f"### {nombre}\n")
            out.append("| # | Ticker | RS | RS Mom | Flujo (z) | WLS | Fase Wyckoff |\n")
            out.append("|---|--------|----|--------|-----------|-----|---------------|\n")
            for i, (_, row) in enumerate(top5.iterrows(), 1):
                out.append(f"| {i} | {row['ticker']} | {_fmt_num(row['rs'], '{:.2f}')} | {_fmt_num(row['rs_mom'], '{:.2%}')} | {_fmt_num(row['flow_proxy_z'], '{:.2f}')} | {_fmt_num(row['wls'], '{:.2f}')} | {row['wyckoff_phase']} |\n")
            out.append("\n")
    else:
        out.append("*Ningún indice en fase de acumulación o markup en esta ejecución.*\n\n")
    return out


def render_sintesis_senales(macro_regime, financial_regime,
                            sector_dispersion_data, breadth_values,
                            price_flow_divergences):
    """Renderiza la seccion Estado Actual - Sintesis de Senales.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    out.append("## Estado Actual — Síntesis de Señales\n\n")

    resumen = []

    # 1. Régimen macro (prioridad máxima)
    if macro_regime in ('RECESSION', 'LIQUIDITY CRISIS', 'STAGFLATION'):
        resumen.append(f"- **Régimen macro: {macro_regime}** — entorno de estrés elevado.")
    elif macro_regime in ('EXPANSION', 'RECOVERY', 'GOLDILOCKS'):
        resumen.append(f"- **Régimen macro: {macro_regime}** — favorable para la asunción de riesgo.")
    elif macro_regime == 'MIXED':
        # Leer dispersion real del ultimo dia (fix C20: texto dinamico)
        disp_txt = 'variable'
        try:
            if sector_dispersion_data is not None and not sector_dispersion_data.empty:
                last = sector_dispersion_data.iloc[-1]
                lectura = last.get('dispersion_reading') or last.get('Lectura')
                if lectura and isinstance(lectura, str):
                    disp_txt = lectura.lower()
        except (KeyError, IndexError, ValueError, TypeError, AttributeError) as e:
            print(f"  [WARN] report_generator: sector_dispersion_data: {e}")
        resumen.append(f"- **Régimen macro: MIXED** — ROTATIONAL / MIXED — rotación sectorial activa con dispersión {disp_txt}.")
    else:
        resumen.append(f"- **Régimen macro: {macro_regime}**.")

    # 3. Condiciones financieras o liquidez (si es relevante)
    if financial_regime in ('HIGH_STRESS', 'EXTREME_STRESS'):
        resumen.append(f"- **Condiciones financieras: {financial_regime}** — condiciones financieras elevadas. El crédito se reporta por separado en Alertas de Divergencia (HYG/LQD). La liquidez real (FRED) se reporta en el Resumen de Regímenes.")
    elif financial_regime == 'ESTRECHA':
        resumen.append("- **Condiciones financieras: ESTRECHA** — señales financieras en territorio restrictivo.")

    # Máximo 3 elementos
    for item in resumen[:3]:
        out.append(item + "\n")
    out.append("\n")

    # Divergencias relevantes (solo si no están ya en el resumen)
    divergencias = []
    if breadth_values:
        ema200 = breadth_values.get('% sobre EMA200', 0)
        ema20 = breadth_values.get('% sobre EMA20', 0)
        if ema200 > 0.70 and ema20 < 0.60:
            divergencias.append(f"- **Breadth Divergence:** Breadth EMA200: {ema200:.0%}; Breadth EMA20: {ema20:.0%}. La amplitud de corto plazo es inferior a la de largo plazo.")
    if price_flow_divergences:
        for ticker, div in price_flow_divergences.items():
            if div.get('status') == 'PRICE_STRONG_FLOW_UNCONFIRMED':
                name = SECTOR_NAMES.get(ticker, ticker)
                divergencias.append(f"- **{name}**: precio fuerte sin confirmación del Flow Proxy.")
    if divergencias and len(resumen) < 3:
        out.append("### Divergencias Relevantes (Síntesis)\n")
        for d in divergencias[:2]:
            out.append(d + "\n")
        out.append("\n")
    # Nota de cierre
    out.append("*Esta sección describe únicamente estados observables del sistema. No interpreta causas ni sugiere acciones.*\n\n")
    
    out.append("\n*Esta interpretacion es descriptiva y no constituye una recomendacion de inversion.*\n\n")
    return out


def render_matriz_evidencia(evidence_matrix_data):
    """Renderiza la tabla Matriz de Evidencia.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if evidence_matrix_data is not None and not evidence_matrix_data.empty:
        def _fmt_evidence(v):
            if pd.isna(v):
                return 'NA'
            # FU-003a (2026-09-15): cero sin signo.
            iv = int(v)
            return f"{iv:+d}" if iv != 0 else "0"

        # F6-29a (2026-09-28): WONT FIX razonado. El valor de la A/D Line
        # acumulada es un contador descriptivo del historico del radar
        # (suma neta de avanzantes - descensos). No tiene "referencia
        # historica" porque el propio valor ES la referencia acumulada.
        # F6-30 (2026-09-28): WONT FIX razonado. Que ningun sector
        # aparezca con "Lectura MIXTA" es consecuencia de los umbrales
        # de alignment_reading: exige n_positivos >= 3 y n_positivos >
        # n_negativos (o simetrico). Sectores con 2-2-1 caen en MIXTA.
        # En regimenes homogeneos, la mayoria cae en FAVORABLE o
        # DESFAVORABLE, no en MIXTA.
        out.append("\n## Matriz de Evidencia\n\n")
        # H13 (2026-10-16, auditoria): el nombre "Flujo 1º" no declaraba
        # el horizonte. Se cambia a "Flujo 1º 20d" y se anade nota
        # metodologica. La columna se calcula sobre primary_flow_evidence
        # (flow_20d_sum).
        out.append("| Sector | Precio | Amplitud | Flujo 1º 20d | Flujo Proxy | Wyckoff | Crédito* | Volat* | Calidad | Votos P/N/0 | Lectura |\n")
        out.append("|--------|--------|----------|----------|-------------|---------|----------|--------|---------|-------------|----------|\n")
        for _, row in evidence_matrix_data.iterrows():
            _pos = row.get('n_sector_evidence_positive', 0)
            _neg = row.get('n_sector_evidence_negative', 0)
            _neu = row.get('n_sector_evidence_neutral', 0)
            _pos = int(_pos) if pd.notna(_pos) else 0
            _neg = int(_neg) if pd.notna(_neg) else 0
            _neu = int(_neu) if pd.notna(_neu) else 0
            out.append(
                f"| {row['sector']} | {_fmt_evidence(row['price_evidence'])} | {_fmt_evidence(row['breadth_evidence'])} | {_fmt_evidence(row['primary_flow_evidence'])} | {_fmt_evidence(row['proxy_flow_evidence'])} | {_fmt_evidence(row['wyckoff_evidence'])} | {_fmt_evidence(row['credit_evidence'])} | {_fmt_evidence(row['volatility_evidence'])} | {row['evidence_quality']} | {_pos}/{_neg}/{_neu} | {row['alignment_reading']} |\n"
            )
        out.append("\n*H13 auditoria 2026-10-16: la columna Flujo 1º 20d "
                   "refleja el signo del flujo primario acumulado a 20 "
                   "sesiones (flow_20d_sum de sector_flow_characteristics). "
                   "Umbral de signo: +1 si > 0, -1 si < 0, 0 si neutro. "
                   "No es el flujo de la ultima sesion.*\n")
        out.append("\n*Crédito y volatilidad representan contexto común de mercado y no participan en el balance de evidencia sectorial.*\n")
        # F6-34 aspecto 1 (2026-09-28): WONT FIX razonado. Que Wyckoff,
        # Credito y Volat aparezcan constantes entre sectores es esperado:
        # - Credito y Volat son contexto global (mismo valor para los 11
        #   sectores por diseno).
        # - Wyckoff depende de pct_accumulation+markup vs distribution+
        #   markdown; en regimenes alcistas los 11 suelen coincidir en +1.
        out.append("*Credito y Volat son contexto global "
                   "(mismo valor para los 11 sectores por diseno).*\n\n")
        # REP-AUDIT-20261020-REPORT-FLOW-RECONCILIATION (P1 auditor externo
        # 2026-10-20): aclarar que la columna Flujo Proxy de esta matriz
        # NO es la misma metrica que la tabla 'Flujo de Mercado - Sectores
        # (Proxy)'. La matriz usa sector_concentration.flow_median
        # (mediana del composite de los componentes del sector).
        out.append("*Nota: la columna 'Flujo Proxy' de esta matriz usa la "
                   "mediana del composite de los componentes del sector "
                   "(sector_concentration.flow_median). La tabla 'Flujo de "
                   "Mercado - Sectores (Proxy)' usa el composite momentum "
                   "del ETF sectorial. Son metricas distintas y pueden "
                   "discrepar en signo.*\n\n")
        # B3 (2026-10-13): explicitar que 3 columnas pueden ser constantes.
        # H4 (2026-10-16, auditoria): el texto afirmaba "los 11 sectores
        # coinciden" en Wyckoff. Empiricamente puede haber excepciones
        # puntuales (ej. XLU=0 en un snapshot). Se reformula sin
        # afirmar uniformidad total.
        out.append("*Columnas con frecuencia constante entre sectores: "
                   "Credit* y Volat* (contexto global, mismo valor para "
                   "los 11). Wyckoff tiende a +1 en regimenes alcistas; "
                   "pueden aparecer excepciones puntuales por sector. La "
                   "variabilidad sectorial real proviene de Precio, "
                   "Amplitud, Flujo 1 y Flujo Proxy.*\n\n")
        # REP-AUDIT-20261020-REPORT-REVIEW-EXTERNAL (auditor externo
        # 2026-10-20): explicitar la regla exacta del voto Wyckoff.
        out.append("*Regla del voto Wyckoff: +1 si "
                   "(pct_accumulation + pct_markup) > "
                   "(pct_distribution + pct_markdown); -1 si <; 0 si =. "
                   "Es un signo agregado del sector, no la fase "
                   "individual de cada ETF. En regimenes alcistas los "
                   f"{EXPECTED_SECTOR_COUNT} sectores pueden coincidir en +1 "
                   "sin que sea un error de calculo.*\n\n")
        # N4 (2026-10-13, auditor) + H9 (2026-10-15, auditor):
        # la agregacion es heuristica de signos (1 voto por variable).
        # H9 pide CUANTIFICAR la doble contabilizacion. Medicion empirica
        # sobre outputs/history/evidence_matrix.csv (24 filas con 5
        # evidencias validas, Pearson):
        #   price <-> breadth:        r = 0.641   <- unico par material
        #   price <-> primary_flow:   r = 0.120
        #   price <-> proxy_flow:     r = 0.015
        #   primary_flow <-> proxy:   r = 0.130
        #   wyckoff <-> resto:        |r| < 0.29
        # Conclusion: de los 10 pares posibles, solo 1 (price-breadth)
        # presenta correlacion material. La sobre-representacion del
        # precio en el balance es acotada. No hay justificacion
        # estadistica para modificar el algoritmo.
        # H18 (2026-10-16, auditoria): la nota previa no documentaba
        # muestra, ventana ni periodo. Se completa. La correlacion
        # NO prueba independencia estadistica general; describe la
        # evidencia disponible en un snapshot concreto.
        out.append("*H9 auditoria 2026-10-15 + H18 auditoria 2026-10-16: "
                   "el balance es una heuristica de signos (1 voto por "
                   "variable). Medicion empirica documentada: fuente "
                   "outputs/history/evidence_matrix.csv, 24 filas con 5 "
                   "evidencias validas (sectores con las 5 columnas "
                   "pobladas), correlacion Pearson contemporanea "
                   "(misma sesion) sobre toda la ventana historica "
                   "disponible en el momento del analisis. "
                   "Resultado: solo el par Precio-Amplitud tiene "
                   "correlacion material (r=0.64); el resto de los 10 "
                   "pares |r|<0.3. El Flow Proxy NO correlaciona con "
                   "Precio (r=0.015) en esa muestra. La etiqueta "
                   "describe confluencia de signos, no confirmacion "
                   "estadistica independiente general. Puede existir "
                   "dependencia por construccion que una correlacion "
                   "puntual no detecta. Ver clasificacion de evidencia "
                   "(OBSERVABLE / DERIVADA-DIRECTA / DERIVADA-COMPUESTA) "
                   "mas abajo.*\n\n")
        out.append("*N4 auditoria 2026-10-13: el balance de evidencia es una "
                   "heuristica de signos (1 voto por variable). Precio, "
                   "Amplitud y Flow Proxy comparten informacion del precio: "
                   "un mismo movimiento puede reflejarse en varias a la vez. "
                   "La etiqueta describe confluencia de signos, no "
                   "confirmacion estadistica independiente. Ver clasificacion "
                   "de evidencia (OBSERVABLE / DERIVADA-DIRECTA / "
                   "DERIVADA-COMPUESTA) en la seccion Anti-Double-Counting.*\n\n")
        out.append("\n*La ausencia de Flow Proxy (NaN) representa ausencia de evidencia disponible y no se interpreta como neutralidad.*\n")
        out.append("\n")
    return out
