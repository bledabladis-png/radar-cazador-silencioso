# -*- coding: utf-8 -*-
"""Seccion IAE (Institutional Accumulation Engine) para el reporte diario.

Renderiza el resultado de compute_iae_section como lista de lineas
markdown. Sin side effects.

Referencia documental: IAE_MAESTRO.md secciones §10.3, §12.5, §13.5.
"""
from __future__ import annotations


def _fmt_int(v):
    if v is None:
        return "N/D"
    try:
        return f"{int(v):,}".replace(",", ".")
    except (TypeError, ValueError):
        return str(v)


def render_iae_section(iae_section):
    """Renderiza la seccion IAE. Devuelve list[str] (markdown).

    Si iae_section is None -> no renderiza nada (devuelve []).
    Si status == STALE -> seccion minima informativa.
    Si status == ERROR -> seccion con aviso de error.
    Si status == OK -> seccion completa.
    """
    if not iae_section:
        return []

    out = []
    out.append("## Acumulacion Institucional (13F)\n")
    out.append("*Cambio neto de posiciones reportadas institucionales, "
               "segun filings SEC 13F.*\n\n")

    status = iae_section.get("status")

    if status == "STALE":
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
    out.append(f"| Total | {_fmt_int(iae_section.get('nipc_total'))} |\n")
    out.append(f"| SOLE  | {_fmt_int(iae_section.get('nipc_sole'))} |\n")
    out.append(f"| DFND  | {_fmt_int(iae_section.get('nipc_dfnd'))} |\n")
    out.append(f"| OTR   | {_fmt_int(iae_section.get('nipc_otr'))} |\n")
    out.append("\n")

    # Cobertura
    out.append("### Cobertura del catalogo radar\n")
    obs = iae_section.get("catalog_keys_observed")
    tot = iae_section.get("catalog_keys_total")
    cov = iae_section.get("catalog_coverage_declared")
    cov_s = f"{cov:.2%}" if isinstance(cov, (int, float)) else "N/D"
    out.append(f"- Cobertura observable: **{obs} / {tot} ({cov_s})**\n")
    out.append(f"- TARGET construido: **{obs}/{obs} (100%, por construccion)**\n")
    out.append(f"- Estado contractual: "
               f"{iae_section.get('coverage_status') or 'N/D'}\n")
    out.append(f"- Calidad de cobertura: "
               f"{iae_section.get('coverage_quality') or 'N/D'}\n")
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