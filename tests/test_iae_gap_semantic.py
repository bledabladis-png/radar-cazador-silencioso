"""Tests de caracterizacion para 2 funciones publicas sin cobertura semantica.

Detectadas el 2026-09-22 por script de cobertura semantica (funciones
publicas sin invocacion en ningun test):
  - relationships.compute_edge_metrics
  - security_type.classify_title_of_class
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.institutional_accumulation.sec_13f.identity import relationships as rel
from src.institutional_accumulation import security_type as st


# ============================================================
# compute_edge_metrics (relationships.py)
# ============================================================


def _edges_df(statuses):
    return pd.DataFrame({
        "ACCESSION_NUMBER": ["ACC" + str(i) for i in range(len(statuses))],
        "INFOTABLE_SK": [str(i) for i in range(len(statuses))],
        "reference_status": statuses,
    })


def test_compute_edge_metrics_vacio():
    df = pd.DataFrame({
        "ACCESSION_NUMBER": [],
        "INFOTABLE_SK": [],
        "reference_status": [],
    })
    m = rel.compute_edge_metrics(df)
    assert m["edge_count_total"] == 0
    assert m["resolution_rate"] == 0.0
    assert m["resolved_edges"] == 0
    assert m["unmapped_missing_om2"] == 0
    assert "source_line_count" not in m  # sin infotable no se calcula


def test_compute_edge_metrics_cuenta_por_status():
    statuses = [
        rel.STATUS_RESOLVED,
        rel.STATUS_RESOLVED,
        rel.STATUS_RESOLVED,
        rel.STATUS_UNMAPPED_MISSING_OM2,
        rel.STATUS_INVALID_NONNUMERIC,
        rel.STATUS_INVALID_ZERO,
        rel.STATUS_INVALID_OUT_OF_DOMAIN,
        rel.STATUS_NO_REFERENCE,
    ]
    m = rel.compute_edge_metrics(_edges_df(statuses))
    assert m["edge_count_total"] == 8
    assert m["resolved_edges"] == 3
    assert m["unmapped_missing_om2"] == 1
    assert m["invalid_reference_non_numeric"] == 1
    assert m["invalid_reference_zero"] == 1
    assert m["invalid_out_of_domain"] == 1
    assert m["no_reference"] == 1
    # denominator = resolved + unmapped = 4; resolution_rate = 3/4
    assert m["resolution_rate"] == 0.75


def test_compute_edge_metrics_denominador_cero():
    statuses = [rel.STATUS_INVALID_ZERO, rel.STATUS_NO_REFERENCE]
    m = rel.compute_edge_metrics(_edges_df(statuses))
    assert m["resolution_rate"] == 0.0


def test_compute_edge_metrics_con_infotable_calcula_source_lines():
    edges = pd.DataFrame({
        "ACCESSION_NUMBER": ["ACC1", "ACC1", "ACC2"],
        "INFOTABLE_SK": ["1", "2", "3"],
        "reference_status": [rel.STATUS_RESOLVED] * 3,
    })
    info = pd.DataFrame({
        "ACCESSION_NUMBER": ["ACC1", "ACC1", "ACC2"],
        "INFOTABLE_SK": ["1", "2", "3"],
    })
    m = rel.compute_edge_metrics(edges, info)
    assert m["source_line_count"] == 3
    assert m["unique_source_line_id"] == 3
    assert m["invalid_source_line_edges"] == 0


def test_compute_edge_metrics_source_line_invalida():
    edges = pd.DataFrame({
        "ACCESSION_NUMBER": ["ACC1", "ACC9"],
        "INFOTABLE_SK": ["1", "999"],
        "reference_status": [rel.STATUS_RESOLVED] * 2,
    })
    info = pd.DataFrame({
        "ACCESSION_NUMBER": ["ACC1"],
        "INFOTABLE_SK": ["1"],
    })
    m = rel.compute_edge_metrics(edges, info)
    assert m["invalid_source_line_edges"] == 1


# ============================================================
# classify_title_of_class (security_type.py)
# ============================================================


@pytest.mark.parametrize("title,esperado", [
    # NON_EQUITY token
    ("PREFERRED STOCK", (st.SECURITY_TYPE_NON_EQUITY, st.STATUS_RESOLVED_NON_EQUITY)),
    ("W EXP", (st.SECURITY_TYPE_NON_EQUITY, st.STATUS_RESOLVED_NON_EQUITY)),
    # NON_EQUITY phrase (CONVERTIBLE BOND)
    ("CONVERTIBLE BOND", (st.SECURITY_TYPE_NON_EQUITY, st.STATUS_RESOLVED_NON_EQUITY)),
    # Fondos
    ("CLOSED END FUND", (st.SECURITY_TYPE_NON_EQUITY, st.STATUS_RESOLVED_NON_EQUITY)),
    ("MUTUAL FUNDS", (st.SECURITY_TYPE_NON_EQUITY, st.STATUS_RESOLVED_NON_EQUITY)),
    ("MONEY MARKET", (st.SECURITY_TYPE_NON_EQUITY, st.STATUS_RESOLVED_NON_EQUITY)),
    # EQUITY anchor
    ("COM", (st.SECURITY_TYPE_EQUITY, st.STATUS_RESOLVED_EQUITY)),
    ("COMMON", (st.SECURITY_TYPE_EQUITY, st.STATUS_RESOLVED_EQUITY)),
    ("ORD", (st.SECURITY_TYPE_EQUITY, st.STATUS_RESOLVED_EQUITY)),
    ("SHS", (st.SECURITY_TYPE_EQUITY, st.STATUS_RESOLVED_EQUITY)),
    # EQUITY phrase
    ("COMMON STOCK", (st.SECURITY_TYPE_EQUITY, st.STATUS_RESOLVED_EQUITY)),
    ("CL A", (st.SECURITY_TYPE_EQUITY, st.STATUS_RESOLVED_EQUITY)),
    ("SPON ADS", (st.SECURITY_TYPE_EQUITY, st.STATUS_RESOLVED_EQUITY)),
    # EQUITY exact
    ("EQUITY", (st.SECURITY_TYPE_EQUITY, st.STATUS_RESOLVED_EQUITY)),
    ("STOCK", (st.SECURITY_TYPE_EQUITY, st.STATUS_RESOLVED_EQUITY)),
    # Fallback
    ("XYZ", (st.SECURITY_TYPE_UNKNOWN, st.STATUS_UNRESOLVED)),
    ("", (st.SECURITY_TYPE_UNKNOWN, st.STATUS_UNRESOLVED)),
    (None, (st.SECURITY_TYPE_UNKNOWN, st.STATUS_UNRESOLVED)),
])
def test_classify_title_of_class_caracterizacion(title, esperado):
    """Spec 3.3 + dictamen #61 Nivel B. Precedencia: NON_EQUITY > fondo
    excluido > EQUITY anchor > UNRESOLVED.
    """
    assert st.classify_title_of_class(title) == esperado


def test_classify_title_of_class_precedencia_non_equity_sobre_equity():
    """NON_EQUITY tiene precedencia sobre EQUITY anchor."""
    # 'COMMON' es anchor EQUITY, pero 'PREFERRED STOCK' contiene token NON_EQUITY.
    r = st.classify_title_of_class("PREFERRED STOCK COMMON")
    assert r == (st.SECURITY_TYPE_NON_EQUITY, st.STATUS_RESOLVED_NON_EQUITY)


def test_classify_title_of_class_no_detecta_conflict():
    """Docstring: esta funcion NO detecta CONFLICT. Solo resolve_security_type."""
    # CONFLICT solo aparece via resolver con evidencia externa.
    # Aqui verificamos que una entrada ambigua cae en UNRESOLVED, no CONFLICT.
    r = st.classify_title_of_class("XYZ")
    assert r[1] != st.STATUS_CONFLICT
    assert r[1] == st.STATUS_UNRESOLVED
