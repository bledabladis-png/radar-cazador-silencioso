# -*- coding: utf-8 -*-
"""Fase 13 del pipeline: IAE (Institutional Accumulation Engine) - 13F.

Calcula el NIPC contractual sobre el ultimo par de trimestres 13F
disponibles y devuelve un dict listo para el reporte.

- No aborta run.py. Un fallo del IAE no debe bloquear el radar diario
  (13F tiene frecuencia trimestral y disponibilidad distinta).
- run.py lo wirea como seccion adicional; el IAE Gate es independiente
  del Validation Gate 10/10.
- Si no hay >=2 trimestres disponibles, devuelve status="STALE".

Pipeline consumido (no modificado):
  src.institutional_accumulation.pipeline_contractual.run_contractual_nipc
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(r"D:\Macro_Sectorial")
DATA_DIR = ROOT / "data" / "sec_13f" / "processed"

_QUARTER_RE = re.compile(r"^(20\d\d)Q([1-4])$")
_QUARTER_END = {"Q1": "03-31", "Q2": "06-30", "Q3": "09-30", "Q4": "12-31"}
COVERAGE_WARNING_THRESHOLD = 0.90


def _list_available_quarters():
    """Devuelve lista ordenada [(folder, iso_period_end), ...] de parquets.

    Ejemplo: [("2025Q4", "2025-12-31"), ("2026Q1", "2026-03-31")]
    """
    if not DATA_DIR.exists():
        return []
    out = []
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        m = _QUARTER_RE.match(d.name)
        if not m:
            continue
        # Sólo cuenta si tiene INFOTABLE.parquet
        if not (d / "INFOTABLE.parquet").exists():
            continue
        year, q = m.group(1), "Q" + m.group(2)
        iso = year + "-" + _QUARTER_END[q]
        out.append((d.name, iso, year + q))
    out.sort(key=lambda x: x[2])
    return [(f, i) for f, i, _ in out]

def compute_iae_section(reference_date, run_id, *, official_dir=None):
    """Calcula la seccion IAE para el reporte diario.

    reference_date: datetime del run (para logs).
    run_id: str del run (para logs).
    official_dir: override del directorio Official List 13(f).

    Returns:
        dict con keys:
            status (str): OK | STALE | ERROR
            period_previous (str): p.ej. "2025Q4"
            period_current (str): p.ej. "2026Q1"
            iso_previous (str): p.ej. "2025-12-31"
            iso_current (str): p.ej. "2026-03-31"
            nipc_total, nipc_sole, nipc_dfnd, nipc_otr (float)
            n_delta_observable (int)
            n_both, n_new, n_exit, n_unresolved_identity (int)
            coverage_status (str): VALID | UNAVAILABLE
            coverage_quality (str): COMPLETE | PARTIAL | UNAVAILABLE
            catalog_coverage_declared (float): observable/total
            catalog_keys_total (int)
            catalog_keys_observed (int)
            evidence_class (str): CONTRACTUAL
            stale_reason (str | None): insufficient_quarters | official_list_pending
            error (str | None)
    """
    print("Calculando seccion IAE (13F - NIPC contractual)...")

    quarters = _list_available_quarters()
    print("  Trimestres disponibles: " + str([q[0] for q in quarters]))

    if len(quarters) < 2:
        print("  Sin suficientes trimestres 13F para calcular delta.")
        return {
            "status": "STALE",
            "stale_reason": "insufficient_quarters",
            "catalog_coverage_warning": None,
            "period_previous": quarters[0][0] if quarters else None,
            "period_current": None,
            "iso_previous": quarters[0][1] if quarters else None,
            "iso_current": None,
            "nipc_total": None, "nipc_sole": None,
            "nipc_dfnd": None, "nipc_otr": None,
            "n_delta_observable": 0,
            "n_both": 0, "n_new": 0, "n_exit": 0,
            "n_unresolved_identity": 0,
            "coverage_status": "UNAVAILABLE",
            "coverage_quality": "UNAVAILABLE",
            "catalog_coverage_declared": None,
            "catalog_keys_total": None,
            "catalog_keys_observed": None,
            "evidence_class": None,
            "error": None,
        }

    f_prev, iso_prev = quarters[-2]
    f_curr, iso_curr = quarters[-1]

    try:
        from src.institutional_accumulation.pipeline_contractual import (
            run_contractual_nipc)
        kwargs = {}
        if official_dir is not None:
            kwargs["official_dir"] = official_dir
        r = run_contractual_nipc(
            folders=(f_prev, f_curr),
            periods=(iso_prev, iso_curr),
            **kwargs)
    except FileNotFoundError as e:
        msg = str(e)
        if "Official List" in msg:
            print("  STALE en IAE (Official List pendiente): " + msg)
            return {
                "status": "STALE",
                "stale_reason": "official_list_pending",
                "catalog_coverage_warning": None,
                "period_previous": f_prev, "period_current": f_curr,
                "iso_previous": iso_prev, "iso_current": iso_curr,
                "nipc_total": None, "nipc_sole": None,
                "nipc_dfnd": None, "nipc_otr": None,
                "n_delta_observable": 0,
                "n_both": 0, "n_new": 0, "n_exit": 0,
                "n_unresolved_identity": 0,
                "coverage_status": "UNAVAILABLE",
                "coverage_quality": "UNAVAILABLE",
                "catalog_coverage_declared": None,
                "catalog_keys_total": None,
                "catalog_keys_observed": None,
                "evidence_class": None,
                "error": msg,
            }
        raise
    except Exception as e:
        print("  ERROR en IAE: " + str(type(e).__name__) + ": " + str(e))
        return {
            "status": "ERROR",
            "stale_reason": None,
            "catalog_coverage_warning": None,
            "period_previous": f_prev, "period_current": f_curr,
            "iso_previous": iso_prev, "iso_current": iso_curr,
            "nipc_total": None, "nipc_sole": None,
            "nipc_dfnd": None, "nipc_otr": None,
            "n_delta_observable": 0,
            "n_both": 0, "n_new": 0, "n_exit": 0,
            "n_unresolved_identity": 0,
            "coverage_status": "UNAVAILABLE",
            "coverage_quality": "UNAVAILABLE",
            "catalog_coverage_declared": None,
            "catalog_keys_total": None,
            "catalog_keys_observed": None,
            "evidence_class": None,
            "error": str(type(e).__name__) + ": " + str(e),
        }

    # Cobertura del catalogo: observadas / total
    cat_total = int(r.get("catalog_keys_total") or 242)
    cat_obs = max(int(r.get("target_q4_size") or 0),
                  int(r.get("target_q1_size") or 0))
    catalog_cov = (cat_obs / cat_total) if cat_total else None
    catalog_warn = (
        catalog_cov is not None
        and catalog_cov < COVERAGE_WARNING_THRESHOLD
    ) if catalog_cov is not None else None

    print("  Periodo: " + f_prev + " -> " + f_curr)
    print("  NIPC total: " + str(r.get("nipc_total")))
    print("  NIPC SOLE/DFND/OTR: "
          + str(r.get("nipc_sole")) + " / "
          + str(r.get("nipc_dfnd")) + " / "
          + str(r.get("nipc_otr")))
    print("  Cobertura catalogo: " + str(cat_obs) + "/" + str(cat_total)
          + " = " + ("{:.2f}%".format(100 * catalog_cov)
                     if catalog_cov is not None else "N/D"))

    return {
        "status": "OK",
        "stale_reason": None,
        "catalog_coverage_warning": catalog_warn,
        "period_previous": f_prev, "period_current": f_curr,
        "iso_previous": iso_prev, "iso_current": iso_curr,
        "nipc_total": r.get("nipc_total"),
        "nipc_sole": r.get("nipc_sole"),
        "nipc_dfnd": r.get("nipc_dfnd"),
        "nipc_otr": r.get("nipc_otr"),
        "n_delta_observable": int(r.get("n_delta_observable") or 0),
        "n_both": int(r.get("n_both") or 0),
        "n_new": int(r.get("n_new") or 0),
        "n_exit": int(r.get("n_exit") or 0),
        "n_unresolved_identity": int(r.get("n_unresolved_identity") or 0),
        "coverage_status": r.get("coverage_status"),
        "coverage_quality": r.get("coverage_quality"),
        "catalog_coverage_declared": catalog_cov,
        "catalog_keys_total": cat_total,
        "catalog_keys_observed": cat_obs,
        "evidence_class": r.get("evidence_class"),
        "error": None,
    }