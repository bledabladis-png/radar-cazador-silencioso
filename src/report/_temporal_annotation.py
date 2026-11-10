# -*- coding: utf-8 -*-
"""Anotacion de fecha efectiva por contrato temporal en el reporte.

Helper unico para que cada seccion declare la fecha efectiva de la
fuente que consume. Cierra parcialmente TEMPORAL-EFFECTIVE-DATE-POR-
MODULO (auditoria 2026-11-03, hallazgo 10.1) y FRESHNESS-POR-BLOQUE
(hallazgo 10.3).

Contrato de entrada: temporal_meta (dict de build_temporal_meta con
by_contract[name] = {effective_date, expected_date, lag_days,
coverage, status}). Fuente canonica de resolucion temporal. No
recalcula fechas ni coberturas; solo formatea lo que ya existe.

Uso desde un render:

    from src.report._temporal_annotation import format_effective_date_line
    line = format_effective_date_line(temporal_meta, "SEC_13F")
    if line:
        out.append(line)

Si temporal_meta es None o el contrato no esta, devuelve "" (el
render decide si publica la linea o no).
"""
from __future__ import annotations



def format_effective_date_line(
    temporal_meta,
    contracts,
    *,
    prefix="Datos al",
    include_lag=True,
):
    """Devuelve '*<prefix> YYYY-MM-DD (lag Nd).*' o '' si no hay contrato.

    Args:
        temporal_meta: dict de build_temporal_meta (o None).
        contracts: str (un contrato) o secuencia de contratos. Si varios,
            se aplica la interseccion efectiva de get_effective_meta
            (min effective_date, min coverage, peor status).
        prefix: texto antes de la fecha. Default 'Datos al'.
        include_lag: si True y lag_days > 0, anade ' (lag Nd)'.

    Returns:
        str listo para out.append(). Incluye '\n' final. '' si no hay
        evidencia (temporal_meta vacio o contrato ausente).
    """
    if not temporal_meta:
        return ""
    if isinstance(contracts, str):
        contracts = [contracts]
    by_contract = temporal_meta.get("by_contract") or {}
    entries = [by_contract[c] for c in contracts if c in by_contract]
    if not entries:
        return ""
    effs = [e.get("effective_date") for e in entries
            if e.get("effective_date") is not None]
    if not effs:
        return ""
    eff = min(effs)
    if hasattr(eff, "strftime"):
        date_str = eff.strftime("%Y-%m-%d")
    else:
        date_str = str(eff)[:10]
    line = "*{0} {1}".format(prefix, date_str)
    if include_lag:
        lags_matching = []
        lags_all = []
        for e in entries:
            lag = e.get("lag_days")
            if lag is None:
                continue
            try:
                lag_int = int(lag)
            except (TypeError, ValueError):
                continue
            lags_all.append(lag_int)
            if e.get("effective_date") == eff:
                lags_matching.append(lag_int)
        if lags_matching:
            lag_int = max(lags_matching)
        elif lags_all:
            lag_int = max(lags_all)
        else:
            lag_int = 0
        if lag_int > 0:
            line += " (lag {0}d)".format(lag_int)
    line += ".*\n"
    return line
