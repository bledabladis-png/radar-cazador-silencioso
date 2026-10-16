# -*- coding: utf-8 -*-
"""Flujos ETF primarios: SPDR + Caracteristicas + Divergencia.

Extraidos de src/report_generator.py (refactor C1, fase C1-6d3a).
"""

import pandas as pd

from config.settings import ETF_PRIMARY_FLOW_ZSCORE_WINDOW
from src.report.helpers import _fmt_num, _fmt_signed


def render_flujo_spdr(etf_primary_flow_data):
    """Renderiza la tabla de Flujo Primario ETF (SPDR).

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if etf_primary_flow_data is not None and not etf_primary_flow_data.empty:
        out.append("## Flujo Primario ETF (SPDR)\n")
        out.append("| Ticker | NAV | Shares Outstanding | ΔShares | Total Net Assets | Primary Flow $ | Flow % AUM | Flow Z |\n")
        out.append("|--------|-----|---------------------|---------|------------------|----------------|------------|--------|\n")

        def _fmt_flow_z(row):
            """H6/H10 (2026-10-15): si el regimen es SATURATED, muestra el
            z clipado con * y el bruto entre parentesis. Antes se mostraba
            +5.00 sin distinguirlo de un z real de +5.00."""
            _z = row.get('primary_flow_z')
            _raw = row.get('primary_flow_z_raw')
            _reg = row.get('primary_flow_z_regime', '')
            _z_str = _fmt_signed(_z, '{:+.2f}', '{:.2f}')
            if _reg == 'SATURATED' and _raw is not None and pd.notna(_raw):
                return f"{_z_str}* ({_raw:+.2f})"
            return _z_str

        for _, row in etf_primary_flow_data.iterrows():
            _sc = row.get('shares_change')
            _sc_str = _fmt_signed(_sc, '{:+,.0f}', '{:,.0f}') if _sc is not None and pd.notna(_sc) else 'N/D'
            out.append(f"| {row['ticker']} | {row['nav']:.2f} | {row['shares_outstanding']:,.0f} | {_sc_str} | {row['total_net_assets']:,.0f} | {_fmt_signed(row['primary_flow_usd'], '{:+,.2f}', '{:,.2f}')} | {_fmt_signed(row['primary_flow_pct'], '{:+.2f}%', '{:.2f}%')} | {_fmt_flow_z(row)} |\n")
        # O1 (2026-09-18): declarar la fecha efectiva del dataset.
        try:
            _ultima_fecha = str(etf_primary_flow_data["Date"].max())[:10]
        except (KeyError, AttributeError, ValueError, TypeError):
            _ultima_fecha = "N/D"
        out.append(f"\n*Fuente: State Street Global Advisors (SSGA). ETF Primary Flow = ΔShares Outstanding × NAV. Z-score sobre {ETF_PRIMARY_FLOW_ZSCORE_WINDOW} sesiones. Ultima fecha: {_ultima_fecha}. SSGA publica con lag de 1-2 dias; la fecha puede ir por detras del run.*\n\n")
        # H12 (2026-10-15, auditoria externa): aclarar que el flujo es una
        # estimacion operativa (creaciones/reembolsos), no una observacion
        # universal del flujo comprador/vendedor en el mercado secundario.
        out.append("*H12 (2026-10-15, auditor): ETF Primary Flow es una "
                   "ESTIMACION OPERATIVA (variacion de participaciones x NAV), "
                   "no una observacion universal del flujo comprador o "
                   "vendedor en el mercado secundario. La creacion de "
                   "participaciones implica compras subyacentes por el AP, "
                   "pero no identifica quien compro el ETF ni a que precio. "
                   "Se usa como proxy de demanda agregada del vehiculo, no "
                   "como medida del flujo economico del activo.*\n\n")
        # H11 (2026-10-16, auditoria): declarar la fuente de dShares y
        # la fecha de comparacion. El flujo se calcula como dShares x NAV,
        # donde dShares = shares_outstanding[t] - shares_outstanding[t-1].
        out.append("*H11 auditoria 2026-10-16: la columna ΔShares declara "
                   "la variacion de participaciones usada en el calculo "
                   "(shares_outstanding[t] - shares_outstanding[t-1]). "
                   "La fecha de comparacion es la fecha inmediatamente "
                   "anterior con dato disponible. Verificacion de "
                   "plausibilidad: variaciones >5% del AUM requieren "
                   "revision manual de la fuente.*\n\n")
        out.append("*Esta tabla muestra flujo por ETF (dato atomico). "
                   "Para la lectura agregada con persistencia y regimen, ver "
                   "la tabla siguiente: Flujo Primario ETF - Agregado sectorial.*\n\n")
        # B10 (2026-10-13): SSGA devuelve tambien ETFs no sectoriales
        # (p. ej. FEZ = SPDR EURO STOXX 50). Aclaracion explicita.
        _tickers = set(etf_primary_flow_data['ticker'].astype(str).str.upper())
        _no_sectoriales = _tickers - {'XLB','XLC','XLE','XLF','XLI','XLK','XLP',
                                        'XLRE','XLU','XLV','XLY'}
        if _no_sectoriales:
            out.append("*ETFs no sectoriales presentes en la fuente SSGA "
                       "(no alimentan el analisis sectorial): " +
                       ", ".join(sorted(_no_sectoriales)) + ".*\n\n")
        # F6-29b (2026-09-28): aclarar +0.00 vs N/D.
        out.append("*En la columna Primary Flow $: `+0.00` indica flujo calculado exactamente cero (shares outstanding sin cambio); `N/D` indica dato no disponible (NaN en shares o NAV). Son estados distintos.*\n\n")
        # H6/H10 (2026-10-15): el asterisco indica saturacion del clip.
        out.append("*En la columna Flow Z: `*` indica que el z-score bruto supero el clip contractual [-5, +5]. Se muestra el limite (+/-5.00) seguido del bruto entre parentesis. El flujo subyacente es un dato real; la saturacion es del z-score.*\n\n")
    return out


def render_flujo_caracteristicas(sector_flow_characteristics_data):
    """Renderiza la tabla Flujo Primario ETF - Caracteristicas.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if sector_flow_characteristics_data is not None and not sector_flow_characteristics_data.empty:
        latest_date = pd.to_datetime(sector_flow_characteristics_data['date']).max()
        flow_latest = sector_flow_characteristics_data[pd.to_datetime(sector_flow_characteristics_data['date']) == latest_date]
        out.append("## Flujo Primario ETF - Agregado sectorial\n")
        out.append("| Sector | Flujo $ | % AUM | Z | 5d Acum | 20d Acum | Pers 5d | Pers 20d | Ret 20d | Régimen |\n")
        out.append("|--------|---------|-------|----|---------|----------|---------|----------|---------|----------|\n")
        for _, row in flow_latest.iterrows():
            regime = row.get('price_flow_regime', None)
            regime_str = regime if pd.notna(regime) else 'N/D'
            out.append(f"| {row['sector']} | {_fmt_signed(row['flow_dollar'], '{:+,.2f}', '{:,.2f}')} | {_fmt_num(row['flow_pct_aum'], '{:.2f}%')} | {_fmt_num(row['flow_zscore'], '{:.2f}')} | {_fmt_signed(row['flow_5d_sum'], '{:+,.0f}', '{:,.0f}')} | {_fmt_signed(row['flow_20d_sum'], '{:+,.0f}', '{:,.0f}')} | {_fmt_num(row['persistence_5d'], '{:.0%}')} | {_fmt_num(row['persistence_20d'], '{:.0%}')} | {_fmt_num(row['price_ret_20d'], '{:.2%}')} | {regime_str} |\n")
        out.append("\n")
        out.append("*Esta tabla agrega el flujo por sector con metricas derivadas (persistencia, regimen). Es la lectura sectorial de la tabla anterior (dato atomico por ETF).*\n\n")
        # H17 (2026-10-15, auditoria externa): aclarar que Pers 5d/20d aqui
        # es la persistencia del SIGNO DEL FLUJO primario por sector
        # (pos / (pos+neg) sobre la ventana). Rango [0,1]. Distinta de
        # "Persist" del Structural Ranking (persistencia del score
        # sectorial) y de Pers 5d/10d/20d de lideres (retorno positivo
        # del ticker).
        out.append("*Pers 5d / Pers 20d en esta tabla: fraccion de la "
                   "ventana en que el flujo primario del sector fue "
                   "positivo (pos / (pos+neg)). Rango [0,1]. Distinta de "
                   "'Persist' del Structural Ranking (persistencia del "
                   "score sectorial) y de Pers 5d/10d/20d de lideres "
                   "(retorno positivo del ticker).*\n\n")
    return out


def render_divergencia_precio_flujo(sector_flow_characteristics_data):
    """Renderiza la tabla Divergencia Precio-Flujo Primario.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if sector_flow_characteristics_data is not None and not sector_flow_characteristics_data.empty:
        _flow_filtered = sector_flow_characteristics_data[pd.to_datetime(sector_flow_characteristics_data['date']) == pd.to_datetime(sector_flow_characteristics_data['date']).max()]
        out.append("## Divergencia Precio–Flujo Primario\n")
        out.append("| Sector | Ret 5d | Flujo 5d | Régimen 5d | Ret 20d | Flujo 20d | Régimen 20d |\n")
        out.append("|--------|--------|----------|------------|---------|-----------|-------------|\n")
        for _, row in _flow_filtered.iterrows():
            out.append(f"| {row['sector']} | {row['price_ret_5d']:.2%} | {_fmt_signed(row['flow_5d_sum'], '{:+,.0f}', '{:,.0f}')} | {row['price_flow_regime_5d']} | {row['price_ret_20d']:.2%} | {_fmt_signed(row['flow_20d_sum'], '{:+,.0f}', '{:,.0f}')} | {row['price_flow_regime_20d']} |\n")
        out.append("\n")
        out.append("*«Absorción potencial» describe una configuración de retorno negativo del precio acompañada de flujo primario acumulado positivo. Puede ser compatible con absorción, pero no confirma por sí sola absorción institucional ni establece causalidad.*\n\n")

        # P2.7 (2026-10-13, auditor): la columna 'date' de la fila es
        # la fecha del FLUJO (SSGA, lag T+1/T+2). Los Ret 5d/20d se
        # calculan sobre df_market, que puede tener fecha posterior
        # (12-oct en el run de 13-oct). Ambas ventanas pueden terminar
        # en sesiones distintas. El regimen (Confirmacion/Divergencia/
        # Absorcion) compara precio y flujo de ventanas no alineadas.
        # N5 (2026-10-13, auditor): decision explicita = mantener la
        # comparacion asincrona, pero declararla con la fecha del flujo
        # visible en el encabezado y con nota reforzada. No se alinean
        # las series a una fecha comun: los retornos de precio se
        # calculan sobre el ultimo cierre disponible (12-oct en el run
        # de 13-oct), el flujo sobre la ultima publicacion SSGA
        # (9-oct). La lectura debe ser de posicion relativa, no de
        # senal contemporanea.
        _flow_date = pd.to_datetime(_flow_filtered['date']).max()
        _flow_date_str = _flow_date.strftime('%Y-%m-%d') if pd.notna(_flow_date) else 'N/D'
        out.append(f"*Fecha del flujo (SSGA): {_flow_date_str}. "
                   "Los retornos 5d/20d usan el ultimo cierre de precio "
                   "disponible (posterior por lag del proveedor).*\n\n")
        out.append("*N5 auditoria 2026-10-13: comparacion asincrona "
                   "declarada. Las series pueden terminar en sesiones "
                   "distintas. La lectura de divergencia precio-flujo es "
                   "de posicion relativa, no de senal contemporanea. No "
                   "usar como confirmacion en tiempo real sin considerar "
                   "el desfase.*\n\n")
    return out
