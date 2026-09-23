"""Tests unitarios de src/report/iae.py (E.4).

Verifican el render markdown de la seccion IAE para los 4 paths:
None, STALE, ERROR, OK.
"""
from __future__ import annotations

from src.report.iae import render_iae_section


def test_none_devuelve_lista_vacia():
    assert render_iae_section(None) == []
    assert render_iae_section({}) == []


def test_stale_render_minimo():
    r = render_iae_section({
        "status": "STALE",
        "period_previous": "2025Q1",
        "period_current": None,
    })
    txt = "".join(r)
    assert "Acumulacion Institucional" in txt
    assert "Sin suficientes trimestres" in txt
    assert "2025Q1" in txt


def test_error_muestra_aviso():
    r = render_iae_section({
        "status": "ERROR",
        "error": "RuntimeError: boom",
    })
    txt = "".join(r)
    assert "No disponible" in txt
    assert "boom" in txt


def test_ok_incluye_nipc_y_cobertura():
    r = render_iae_section({
        "status": "OK",
        "period_previous": "2025Q4",
        "period_current": "2026Q1",
        "nipc_total": -4316734936.0,
        "nipc_sole": -32484713330.0,
        "nipc_dfnd":  28317652408.0,
        "nipc_otr":   -149674014.0,
        "n_delta_observable": 553321,
        "n_both": 444276, "n_new": 60324, "n_exit": 48721,
        "n_unresolved_identity": 0,
        "coverage_status": "VALID",
        "coverage_quality": "COMPLETE",
        "catalog_keys_observed": 240,
        "catalog_keys_total": 242,
        "catalog_coverage_declared": 240/242,
        "evidence_class": "CONTRACTUAL",
    })
    txt = "".join(r)
    assert "2025Q4 -> 2026Q1" in txt
    assert "NIPC" in txt
    assert "4.316.734.936" in txt
    assert "240 / 242" in txt
    assert "99,17%" in txt or "99.17%" in txt
    assert "553.321" in txt
    assert "CONTRACTUAL" not in txt or "Metodo: CONTRACTUAL" in txt


def test_ok_no_confunde_cobertura_con_100():
    r = render_iae_section({
        "status": "OK",
        "period_previous": "2025Q4",
        "period_current": "2026Q1",
        "nipc_total": 0, "nipc_sole": 0, "nipc_dfnd": 0, "nipc_otr": 0,
        "n_delta_observable": 0, "n_both": 0, "n_new": 0, "n_exit": 0,
        "n_unresolved_identity": 0,
        "coverage_status": "VALID", "coverage_quality": "COMPLETE",
        "catalog_keys_observed": 240, "catalog_keys_total": 242,
        "catalog_coverage_declared": 240/242,
    })
    txt = "".join(r)
    # La nota metodologica debe estar presente
    assert "no constituye una" in txt
    assert "estimacion independiente" in txt


def test_status_desconocido():
    r = render_iae_section({"status": "FOO"})
    txt = "".join(r)
    assert "desconocido" in txt.lower() or "FOO" in txt