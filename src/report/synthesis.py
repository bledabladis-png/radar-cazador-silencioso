# -*- coding: utf-8 -*-
"""Índices (USA + Europa), Sintesis de Senales y Matriz de Evidencia.

Extraidos de src/report_generator.py (refactor C1, fase C1-6d6).
"""

import pandas as pd

from config.index_tickers import INDEX_CONFIG
from src.report.helpers import _fmt_num
from config.tickers import SECTOR_NAMES


def render_indices_internacionales(index_phases, index_leaders):
    """Renderiza las tablas de Índices (USA + Europa) (Wyckoff + Oportunidades).

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    out.append("\n## Índices (USA + Europa) — Fases Wyckoff\n")
    out.append("| Indice | Ticker | Fase Wyckoff |\n")
    out.append("|--------|--------|--------------|\n")
    if index_phases:
        for nombre, fase in index_phases.items():
            ticker = INDEX_CONFIG.get(nombre, {}).get('index_ticker', '')
            out.append(f"| {nombre} | {ticker} | {fase} |\n")
    else:
        out.append("| No disponible | No disponible | No disponible |\n")
    out.append("\n")

    out.append("\n## Índices (USA + Europa) — Oportunidades de Acumulación y Markup\n")
    out.append("*Nota: Los componentes se obtienen de ETFs proxy que replican el indice de referencia. USA: SPY (S&P 500), DIA (Dow Jones), QQQ (Nasdaq-100), IWM (Russell 2000). Europa: FEZ (Euro Stoxx 50), LYXI (Ibex 35), DAXEX (DAX 40), ISF.L (FTSE 100). Solo se muestran indices en fase ACCUMULATION o MARKUP.*\n\n")
    out.append("*Nota: Flujo (z) es un z-score robusto sobre 60 días. Valores extremos pueden deberse a eventos corporativos o volúmenes inusuales. Para el WLS se usa una versión normalizada limitada a ±3.*\n\n")
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


def render_sintesis_senales(macro_regime, slpm_v12_data, liquidity_regime,
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

    # 2. Liderazgo sectorial (prioridad alta)
    if slpm_v12_data:
        leader = slpm_v12_data.get('sector', '')
        state = slpm_v12_data.get('state', '')
        if leader and state:
            if state == 'CONFIRMED':
                resumen.append(f"- **Liderazgo confirmado: {leader}** (SLPM: CONFIRMED).")
            elif state == 'UNRESOLVED':
                resumen.append(f"- **Liderazgo no confirmado: {leader}** (#1 del ranking, SLPM: UNRESOLVED).")
            else:
                resumen.append(f"- **Liderazgo sectorial: {leader}** (SLPM: {state}).")

    # 3. Condiciones financieras o liquidez (si es relevante)
    if liquidity_regime in ('HIGH_STRESS', 'EXTREME_STRESS'):
        resumen.append(f"- **Condiciones financieras: {liquidity_regime}** — estrés elevado en crédito y liquidez.")
    elif liquidity_regime == 'ESTRECHA':
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
        out.append("| Sector | Precio | Amplitud | Flujo 1º | Flujo Proxy | Wyckoff | Crédito* | Volat* | Calidad | Lectura |\n")
        out.append("|--------|--------|----------|----------|-------------|---------|----------|--------|---------|----------|\n")
        for _, row in evidence_matrix_data.iterrows():
            out.append(
                f"| {row['sector']} | {_fmt_evidence(row['price_evidence'])} | {_fmt_evidence(row['breadth_evidence'])} | {_fmt_evidence(row['primary_flow_evidence'])} | {_fmt_evidence(row['proxy_flow_evidence'])} | {_fmt_evidence(row['wyckoff_evidence'])} | {_fmt_evidence(row['credit_evidence'])} | {_fmt_evidence(row['volatility_evidence'])} | {row['evidence_quality']} | {row['alignment_reading']} |\n"
            )
        out.append("\n*Crédito y volatilidad representan contexto común de mercado y no participan en el balance de evidencia sectorial.*\n")
        # F6-34 aspecto 1 (2026-09-28): WONT FIX razonado. Que Wyckoff,
        # Credito y Volat aparezcan constantes entre sectores es esperado:
        # - Credito y Volat son contexto global (mismo valor para los 11
        #   sectores por diseno).
        # - Wyckoff depende de pct_accumulation+markup vs distribution+
        #   markdown; en regimenes alcistas los 11 suelen coincidir en +1.
        out.append("*Credito y Volat son contexto global (mismo valor para los 11 sectores por diseno).*\n\n")
        out.append("\n*La ausencia de Flow Proxy (NaN) representa ausencia de evidencia disponible y no se interpreta como neutralidad.*\n")
        out.append("\n")
    return out
