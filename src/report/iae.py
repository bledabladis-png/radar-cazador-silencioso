# -*- coding: utf-8 -*-
"""Seccion IAE (Institutional Accumulation Engine) para el reporte diario.

Renderiza el resultado de compute_iae_section como lista de lineas
markdown. Sin side effects.

Referencia documental: IAE_MAESTRO.md secciones §10.3, §12.5, §13.5.
"""
from __future__ import annotations


def _fmt_int(v):
    # I1 B6 (2026-10-31): pd.notna. NaN -> int(NaN) -> ValueError -> str = nan;
    # inf -> OverflowError no capturado -> crash. Ahora ambos -> N/D.
    import math
    import pandas as pd
    if v is None or not pd.notna(v):
        return "N/D"
    if isinstance(v, float) and not math.isfinite(v):
        return "N/D"
    # H11 (auditoria 2026-11-07): int(v) truncaba hacia cero. Con
    # valores fraccionarios, int(a) + int(b) + int(c) != int(a+b+c)
    # (truncado no es distributivo). El auditor detecto +1 accion de
    # diferencia entre NIPC Total y la suma de breakdowns. int(round)
    # reduce el sesgo individual; el bloque NIPC deriva el Total de
    # la suma de breakdowns redondeados para garantizar consistencia
    # aritmetica por construccion.
    try:
        return f"{int(round(v)):,}".replace(",", ".")
    except (TypeError, ValueError, OverflowError):
        return str(v)


def render_iae_section(iae_section, temporal_meta=None):
    """Renderiza la seccion IAE. Devuelve list[str] (markdown).

    Si iae_section is None -> no renderiza nada (devuelve []).
    Si status == STALE -> seccion minima informativa.
    Si status == ERROR -> seccion con aviso de error.
    Si status == OK -> seccion completa.
    """
    if not iae_section:
        return []

    out = []
    # P2.6 (2026-10-13, auditor): el titulo anterior "Acumulacion
    # Institucional" inducia a leer el NIPC como compras netas. En
    # realidad es variacion neta de posiciones DECLARADAS, no flujo
    # economico (puede variar por precio, no solo por compra).
    out.append("## Variacion neta de posiciones declaradas (13F)\n")
    out.append("*Cambio neto de posiciones reportadas institucionales, "
               "segun filings SEC 13F.*\n\n")
    from src.report._temporal_annotation import format_effective_date_line
    _date_line = format_effective_date_line(temporal_meta, "SEC_13F")
    if _date_line:
        out.append(_date_line)

    status = iae_section.get("status")

    if status == "STALE":
        reason = iae_section.get("stale_reason")
        if reason == "official_list_pending":
            prev = iae_section.get("period_previous")
            curr = iae_section.get("period_current")
            out.append(
                f"**Datos 13F cargados ({prev} -> {curr}), pero la "
                f"Official List 13(f) de {curr} aun no ha sido publicada "
                f"por SEC.**\n"
            )
            out.append(
                "*SEC publica primero el PDF y despues el TXT (a veces con "
                "semanas de retraso). El NIPC se calculara automaticamente "
                "cuando el TXT este disponible.*\n\n"
            )
            return out
        out.append("**Sin suficientes trimestres 13F disponibles para "
                   "calcular variacion.**\n")
        prev = iae_section.get("period_previous")
        if prev:
            out.append(f"*Ultimo trimestre cargado: {prev}*\n")
        out.append("\n")
        return out

    if status == "ERROR":
        err = iae_section.get("error") or "desconocido"
        out.append(f"**No disponible en esta ejecucion.** "
                   f"({err})\n\n")
        return out

    if status != "OK":
        out.append(f"**Estado desconocido: {status}**\n\n")
        return out

    # status == OK
    prev = iae_section.get("period_previous")
    curr = iae_section.get("period_current")
    out.append(f"*Periodo observado: **{prev} -> {curr}***  \n")
    out.append("*Metodo: CONTRACTUAL. NIPC = variacion neta de posiciones "
               "reportadas, no flujo economico.*\n\n")

    # NIPC
    out.append("### NIPC - Cambio neto de posiciones reportadas\n")
    out.append("| Concepto | Valor |\n")
    out.append("|---|---:|\n")
    # H11 (auditoria 2026-11-07): Total se deriva de la suma de los
    # breakdowns redondeados para garantizar que la tabla sea
    # aritmeticamente consistente por construccion. Con floats
    # fraccionarios, int(a)+int(b)+int(c) != int(a+b+c). El bug del
    # auditor (+1 accion) era de presentacion, no de calculo: el
    # compute_nipc raw es correcto (raw_total == sum(raw_parts)).
    def _int_r(v):
        try:
            if v is None:
                return None
            import math
            fv = float(v)
            if not math.isfinite(fv):
                return None
            return int(round(fv))
        except (TypeError, ValueError, OverflowError):
            return None
    _sole_i = _int_r(iae_section.get('nipc_sole'))
    _dfnd_i = _int_r(iae_section.get('nipc_dfnd'))
    _otr_i = _int_r(iae_section.get('nipc_otr'))
    _total_raw_i = _int_r(iae_section.get('nipc_total'))
    _any_breakdown_valid = (
        _sole_i is not None or _dfnd_i is not None or _otr_i is not None)
    _total_derived_i = (
        ((_sole_i if _sole_i is not None else 0)
         + (_dfnd_i if _dfnd_i is not None else 0)
         + (_otr_i if _otr_i is not None else 0))
        if _any_breakdown_valid else None)
    # H11 (auditoria 2026-11-07): si nipc_total es NaN/None (dato
    # faltante), mostrar N/D aunque haya breakdowns. Enmascararlo
    # con un derivado ocultaria ausencia de dato. Solo cuando
    # nipc_total esta presente se deriva Total = suma breakdowns
    # para garantizar consistencia aritmetica con la tabla.
    _total_to_show = _total_derived_i if _total_raw_i is not None else None
    out.append(f"| Total | {_fmt_int(_total_to_show)} |\n")
    out.append(f"| SOLE  | {_fmt_int(_sole_i)} |\n")
    out.append(f"| DFND  | {_fmt_int(_dfnd_i)} |\n")
    out.append(f"| OTR   | {_fmt_int(_otr_i)} |\n")
    # Nota condicional: solo si el Total derivado difiere del raw
    # redondeado (redondeo no distributivo). Cierra la coherencia con
    # el JSON interno sin mentir.
    if (_total_raw_i is not None and _total_derived_i is not None
            and _total_raw_i != _total_derived_i):
        out.append(
            f"*Suma de breakdowns presentada con redondeo consistente. "
            f"El `nipc_total` exacto (raw redondeado: {_total_raw_i:,}) "
            f"puede diferir en +-1 por redondeo independiente.*\n"
            .replace(",", ".")
        )
    out.append("\n")
    # H14 (2026-10-15, auditoria externa): documentar unidad, formula,
    # tratamiento NEW/EXIT, deduplicacion de gestores y significado
    # exacto de COMPLETE.
    out.append("*H14 (2026-10-15, auditor):*\n")
    out.append("*Unidad: ACCIONES (shares), no dolares. NIPC = suma neta "
               "de variacion de numero de acciones declaradas, no variacion "
               "de valor de mercado.*\n")
    out.append("*Formula: NIPC = SUM_i (Shares_i,t - Shares_i,t-1), sobre "
               "reported_position_unit = (periodo, CIK gestor, security "
               "canonica, tipo de discretion).*\n")
    out.append("*Filtro base: solo SSHPRNAMTTYPE='SH' sin PUTCALL "
               "(excluye opciones).*\n")
    out.append("*Discretion: se suman SOLE + DFND + OTR; el breakdown se "
               "preserva sin mezclar.*\n")
    out.append("*NEW asume previous=0 (posicion nueva declarada). EXIT "
               "asume current=0 (posicion desaparecida). UNRESOLVED_IDENTITY "
               "NO contribuye al total (no se fabrica delta).*\n")
    out.append("*Deduplicacion: SSHPRNAMT entra UNA SOLA VEZ por source "
               "line, no se multiplica por edges OTHERMANAGER.*\n")
    out.append("*Naturaleza: GROSS OBSERVED DELTA. NO distingue "
               "compra/venta de split/spin-off/merger (requiere fuente "
               "externa de corporate actions, no disponible en 13F). "
               "Tampoco es flujo economico.*\n\n")

    # FASE-A-IAE-PER-TICKER (2026-11-01): tabla por ticker derivada
    # del DataFrame per_ticker. Solo tickers que pasan filtros de
    # anomalia. Los excluidos van a nota al final, con razon visible.
    # Sin hardcode: los 20 top son parametro; los excluidos dependen
    # del propio DataFrame.
    _per_ticker = iae_section.get("per_ticker")
    if _per_ticker is not None and not _per_ticker.empty:
        from src.institutional_accumulation.aggregation.per_ticker import (
            top_accumulating, top_distributing, excluded_anomalies,
        )
        _top_acc = top_accumulating(_per_ticker, n=20)
        _top_dis = top_distributing(_per_ticker, n=20)
        _excl = excluded_anomalies(_per_ticker)
        if not _top_acc.empty or not _top_dis.empty:
            out.append("### Cambio neto por ticker (13F)\n")
            out.append("*Top por consenso de gestores (cuantos aumentan vs "
                       "cuantos reducen), no por suma bruta de acciones. "
                       "Un ticker con 500 gestores aumentando tiene mas "
                       "consenso que uno con 50 aumentando mas acciones.*\n\n")
            if not _top_acc.empty:
                out.append("**Acumulando** (mas gestores aumentan que reducen)\n\n")
                out.append("| Ticker | Gestores | Aumentan | Reducen | New | Exit | Delta (M acciones) | % aumento |\n")
                out.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
                for _, _r in _top_acc.iterrows():
                    out.append(
                        f"| {_r['ticker']} | {int(_r['n_managers'])} | "
                        f"{int(_r['n_increase'])} | {int(_r['n_decrease'])} | "
                        f"{int(_r['n_new'])} | {int(_r['n_exit'])} | "
                        f"{_fmt_int(_r['sum_delta'] / 1e6)} | "
                        f"{_r['pct_increase']:.2%} |\n"
                    )
                out.append("\n")
            if not _top_dis.empty:
                out.append("**Distribuyendo** (mas gestores reducen que aumentan)\n\n")
                out.append("| Ticker | Gestores | Aumentan | Reducen | New | Exit | Delta (M acciones) | % aumento |\n")
                out.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
                for _, _r in _top_dis.iterrows():
                    out.append(
                        f"| {_r['ticker']} | {int(_r['n_managers'])} | "
                        f"{int(_r['n_increase'])} | {int(_r['n_decrease'])} | "
                        f"{int(_r['n_new'])} | {int(_r['n_exit'])} | "
                        f"{_fmt_int(_r['sum_delta'] / 1e6)} | "
                        f"{_r['pct_increase']:.2%} |\n"
                    )
                out.append("\n")
            if not _excl.empty:
                # Nota derivada: agrupa por filter_reason sin lista de tickers.
                _by_reason = {}
                for _, _r in _excl.iterrows():
                    _by_reason.setdefault(_r["filter_reason"], []).append(_r["ticker"])
                _labels = {
                    "MOSTLY_NEW_POSITIONS": "posiciones nuevas masivas (posible IPO o rebalanceo)",
                    "MOSTLY_EXITS": "salidas masivas (posible spin-off o delisting)",
                    "ANOMALOUS_ONE_SIDED": "movimiento unidireccional >=99% (posible evento corporativo)",
                }
                _parts = []
                for _reason, _tickers in sorted(_by_reason.items()):
                    _label = _labels.get(_reason, _reason)
                    _tlist = ", ".join(sorted(_tickers)[:8])
                    if len(_tickers) > 8:
                        _tlist += "..."
                    _parts.append(f"{len(_tickers)} {_label}: {_tlist}")
                out.append("*Excluidos de las tablas por evento societario o IPO "
                           "(filtros de ratio, no lista de tickers): "
                           + "; ".join(_parts) + ".*\n\n")

    # Cobertura
    out.append("### Cobertura del catalogo radar\n")
    obs = iae_section.get("catalog_keys_observed")
    tot = iae_section.get("catalog_keys_total")
    cov = iae_section.get("catalog_coverage_declared")
    # I2 B6 (2026-10-31): pd.notna. isinstance(NaN, float) = True -> nan%.
    import pandas as _pd
    cov_s = f"{cov:.2%}" if isinstance(cov, (int, float)) and _pd.notna(cov) else "N/D"
    out.append(f"- Cobertura observable: **{obs} / {tot} ({cov_s})**\n")
    if iae_section.get("catalog_coverage_warning"):
        out.append("**[WARN] Cobertura del catalogo por debajo del 90% "
                   "(coverage_quality != COMPLETE).**\n")
    elif (iae_section.get("catalog_coverage_declared") is not None
          and iae_section["catalog_coverage_declared"] < 0.90
          and iae_section.get("coverage_quality") == "COMPLETE"):
        out.append("*[INFO] Cobertura del catalogo por debajo del 90%, "
                   "pero coverage_quality=COMPLETE: el catalogo ha "
                   "crecido mas rapido que el 13F trimestral. No es "
                   "regresion del pipeline (catalogo daily vs 13F "
                   "trimestral).*\n")
    out.append(f"- TARGET construido: **{obs}/{obs} (100%, por construccion)**\n")
    out.append(f"- Estado contractual: "
               f"{iae_section.get('coverage_status') or 'N/D'}\n")
    out.append(f"- Calidad de cobertura: "
               f"{iae_section.get('coverage_quality') or 'N/D'}\n")
    # H14 (2026-10-15): aclarar el significado exacto de los valores.
    out.append("  - *COMPLETE: la cobertura observable alcanza el umbral "
               "interno COVERAGE_COMPLETE_THRESHOLD; no implica cobertura "
               "del 100% del catalogo original.*\n")
    out.append("  - *PARTIAL: cobertura intermedia entre los dos umbrales.*\n")
    out.append("  - *UNRESOLVED: por debajo del umbral minimo o sin "
               "denominador fiable.*\n")
    out.append("\n")
    out.append("*Nota: la cobertura TARGET es interna al universo construido "
               "a partir de observaciones VERIFIED; no constituye una "
               "estimacion independiente de cobertura del catalogo.*\n\n")

    # Observabilidad
    out.append("### Observabilidad\n")
    out.append(f"- Filas observables: "
               f"{_fmt_int(iae_section.get('n_delta_observable'))}\n")
    out.append(f"- BOTH: {_fmt_int(iae_section.get('n_both'))} | "
               f"NEW: {_fmt_int(iae_section.get('n_new'))} | "
               f"EXIT: {_fmt_int(iae_section.get('n_exit'))}\n")
    unres = iae_section.get("n_unresolved_identity")
    if unres:
        out.append(f"- UNRESOLVED_IDENTITY: {_fmt_int(unres)}\n")
    else:
        out.append("- UNRESOLVED_IDENTITY: 0\n")
    out.append("\n")

    return out