"""Test H14: el bloque 13F declara unidad, formula, tratamiento
NEW/EXIT, deduplicacion y significado de COMPLETE.

Auditoria externa 2026-10-15, hallazgo H14 (medio): el reporte muestra
NIPC con desglose SOLE/DFND/OTR pero sin unidad, formula ni tratamiento
de posiciones nuevas/desaparecidas.
"""
from src.report.iae import render_iae_section


def _iae_section():
    return {
        "status": "OK",
        "period_previous": "2026Q1",
        "period_current": "2026Q2",
        "current_quarter": "2026Q2",
        "nipc_total": 8168484363,
        "nipc_sole": 6044894370,
        "nipc_dfnd": 2167136725,
        "nipc_otr": -43546732,
        "catalog_keys_observed": 241,
        "catalog_keys_total": 255,
        "catalog_coverage_declared": 0.9451,
        "catalog_coverage_warning": False,
        "coverage_status": "VALID",
        "coverage_quality": "COMPLETE",
        "n_delta_observable": 584467,
        "n_both": 459994,
        "n_new": 64432,
        "n_exit": 60041,
        "n_unresolved_identity": 0,
    }


def test_h14_declara_unidad_acciones():
    out = "".join(render_iae_section(_iae_section()))
    assert "Unidad: ACCIONES" in out or "shares" in out.lower()
    assert "no dolares" in out.lower() or "no variacion de valor" in out.lower()


def test_h14_declara_formula():
    out = "".join(render_iae_section(_iae_section()))
    assert "NIPC = SUM" in out or "NIPC = sum" in out.lower()


def test_h14_explica_new_exit():
    out = "".join(render_iae_section(_iae_section()))
    assert "NEW asume previous=0" in out
    assert "EXIT asume current=0" in out
    assert "UNRESOLVED_IDENTITY" in out


def test_h14_deduplicacion():
    out = "".join(render_iae_section(_iae_section()))
    assert "UNA SOLA VEZ" in out
    assert "OTHERMANAGER" in out


def test_h14_complete_significado():
    out = "".join(render_iae_section(_iae_section()))
    assert "COMPLETE" in out
    assert "no implica cobertura del 100%" in out or "no implica cobertura" in out.lower()


def test_h14_gross_observed_delta():
    out = "".join(render_iae_section(_iae_section()))
    assert "GROSS OBSERVED DELTA" in out or "gross observed" in out.lower()
