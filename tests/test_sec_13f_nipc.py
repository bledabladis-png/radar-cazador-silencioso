# -*- coding: utf-8 -*-
"""Tests de aggregation.nipc (dictamen NIPC v1.1 familias 12.3-12.6).

Sin red. Deterministas.
"""
import pandas as pd

from src.institutional_accumulation.aggregation import delta_shares as ds
from src.institutional_accumulation.aggregation import nipc as npc


# ---- helpers ----

def _units(rows):
    """Construye units_df a partir de dicts con columnas UNITS_COLUMNS."""
    defaults = {
        "report_period": "2026-03-31",
        "filing_manager_cik": "FM1",
        "observed_security_key": "cusip:C1",
        "security_resolution_status": "CANONICAL",
        "canonical_security_kind": "CANONICAL_EQUIVALENCE",
        "canonical_security": "equity:C1",
        "discretion_type": "SOLE",
        "sshprnamt_total": 100.0,
        "n_source_lines": 1,
    }
    full = [{**defaults, **r} for r in rows]
    return pd.DataFrame(full, columns=list(ds.UNITS_COLUMNS))


def _delta(rows):
    """Construye delta_df a partir de dicts con columnas DELTA_COLUMNS."""
    defaults = {
        "filing_manager_cik": "FM1",
        "canonical_security": "equity:C1",
        "observed_security_key": "cusip:C1",
        "discretion_type": "SOLE",
        "sshprnamt_current": 100.0,
        "sshprnamt_previous": 80.0,
        "delta_shares": 20.0,
        "match_status": ds.STATUS_BOTH,
    }
    full = [{**defaults, **r} for r in rows]
    return pd.DataFrame(full, columns=list(ds.DELTA_COLUMNS))


# ---- constantes ----

def test_status_nipc_definidos():
    assert npc.STATUS_READY == "READY"
    assert npc.STATUS_INSUFFICIENT == "INSUFFICIENT"
    assert npc.STATUS_CONFLICT == "CONFLICT"
    assert npc.STATUS_AMBIGUOUS == "AMBIGUOUS"
    assert npc.STATUS_UNRESOLVED == "UNRESOLVED"
    assert npc.STATUS_TEMPORAL_UNVERIFIED == "TEMPORAL_UNVERIFIED"
    assert len(npc.ALL_NIPC_STATUSES) == 6


def test_discretion_types():
    assert npc.DISCRETION_TYPES == ("SOLE", "DFND", "OTR")


# ---- compute_nipc ----

def test_nipc_total_suma_solo_observable():
    d = _delta([
        {"match_status": ds.STATUS_BOTH, "delta_shares": 20.0},
        {"match_status": ds.STATUS_NEW, "delta_shares": 100.0},
        {"match_status": ds.STATUS_EXIT, "delta_shares": -50.0},
        {"match_status": ds.STATUS_UNRESOLVED_IDENTITY, "delta_shares": None},
    ])
    r = npc.compute_nipc(d)
    assert r["nipc_total"] == 70.0
    assert r["n_both"] == 1
    assert r["n_new"] == 1
    assert r["n_exit"] == 1
    assert r["n_unresolved_identity"] == 1
    assert r["n_delta_observable"] == 3


def test_nipc_breakdown_por_discretion():
    d = _delta([
        {"discretion_type": "SOLE", "delta_shares": 20.0},
        {"discretion_type": "DFND", "delta_shares": 15.0},
        {"discretion_type": "OTR", "delta_shares": -5.0},
    ])
    r = npc.compute_nipc(d, discretion_breakdown=True)
    assert r["nipc_sole"] == 20.0
    assert r["nipc_dfnd"] == 15.0
    assert r["nipc_otr"] == -5.0
    assert r["nipc_total"] == 30.0


def test_nipc_sin_breakdown():
    d = _delta([{"discretion_type": "SOLE", "delta_shares": 20.0}])
    r = npc.compute_nipc(d, discretion_breakdown=False)
    assert "nipc_sole" not in r
    assert "nipc_dfnd" not in r
    assert "nipc_otr" not in r
    assert r["nipc_total"] == 20.0


def test_nipc_vacio():
    r = npc.compute_nipc(pd.DataFrame())
    assert r["nipc_total"] == 0.0
    assert r["n_both"] == 0
    assert r["nipc_sole"] == 0.0
    assert r["nipc_dfnd"] == 0.0
    assert r["nipc_otr"] == 0.0


def test_nipc_unresolved_no_contribuye():
    d = _delta([
        {"match_status": ds.STATUS_UNRESOLVED_IDENTITY, "delta_shares": None},
        {"match_status": ds.STATUS_UNRESOLVED_IDENTITY, "delta_shares": None},
    ])
    r = npc.compute_nipc(d)
    assert r["nipc_total"] == 0.0
    assert r["n_delta_observable"] == 0


# ---- compute_coverage_pairwise ----

def test_coverage_pairwise_ambos_mapeados():
    u_c = _units([
        {"observed_security_key": "cusip:A", "sshprnamt_total": 100.0},
        {"observed_security_key": "cusip:B", "sshprnamt_total": 200.0},
    ])
    u_p = _units([
        {"observed_security_key": "cusip:A", "sshprnamt_total": 80.0},
        {"observed_security_key": "cusip:B", "sshprnamt_total": 150.0},
    ])
    c = npc.compute_coverage_pairwise(u_c, u_p)
    assert c["coverage_current"] == 1.0
    assert c["coverage_previous"] == 1.0
    assert c["paired_security_coverage"] == 1.0
    assert c["paired_weighted_share_coverage"] == 1.0
    assert c["unmapped_weight_current"] == 0.0
    assert c["unmapped_weight_previous"] == 0.0


def test_coverage_pairwise_parcial():
    """P38: denominador = interseccion de TARGET, no union de observadas."""
    u_c = _units([
        {"observed_security_key": "cusip:A"},
        {"observed_security_key": "cusip:B",
         "security_resolution_status": "OBSERVED_ONLY",
         "canonical_security": None},
    ])
    u_p = _units([
        {"observed_security_key": "cusip:A"},
    ])
    c = npc.compute_coverage_pairwise(u_c, u_p)
    assert c["coverage_current"] == 0.5
    assert c["coverage_previous"] == 1.0
    # P38: TARGET_PAIRWISE = {A} (interseccion), both_mapped = {A} -> 1.0
    assert c["paired_security_coverage"] == 1.0


def test_coverage_pairwise_ambos_vacios():
    c = npc.compute_coverage_pairwise(pd.DataFrame(), pd.DataFrame())
    assert c["paired_security_coverage"] == 0.0
    assert c["coverage_current"] == 0.0
    assert c["coverage_previous"] == 0.0


def test_coverage_pairwise_solo_current():
    u_c = _units([{"observed_security_key": "cusip:A"}])
    c = npc.compute_coverage_pairwise(u_c, pd.DataFrame())
    assert c["coverage_current"] == 1.0
    assert c["coverage_previous"] == 0.0
    assert c["paired_security_coverage"] == 0.0


# ---- compute_nipc_and_coverage ----

def test_status_insufficient_sin_thresholds():
    d = _delta([{"match_status": ds.STATUS_BOTH, "delta_shares": 20.0}])
    u_c = _units([{"observed_security_key": "cusip:C1"}])
    u_p = _units([{"observed_security_key": "cusip:C1"}])
    r = npc.compute_nipc_and_coverage(d, u_c, u_p)
    assert r["status"] == npc.STATUS_INSUFFICIENT
    assert r["nipc_total"] == 20.0


def test_status_ready_requiere_ambos_thresholds():
    d = _delta([{"match_status": ds.STATUS_BOTH, "delta_shares": 20.0}])
    u_c = _units([{"observed_security_key": "cusip:C1"}])
    u_p = _units([{"observed_security_key": "cusip:C1"}])
    # thresholds bajos -> ambos pasan
    r = npc.compute_nipc_and_coverage(
        d, u_c, u_p, threshold_1=0.5, threshold_2=0.5,
    )
    assert r["status"] == npc.STATUS_READY


def test_status_no_ready_si_solo_uno_pasa():
    d = _delta([{"match_status": ds.STATUS_BOTH, "delta_shares": 20.0}])
    u_c = _units([{"observed_security_key": "cusip:C1"}])
    u_p = _units([{"observed_security_key": "cusip:C1"}])
    # threshold_1 pasa, threshold_2 no
    r = npc.compute_nipc_and_coverage(
        d, u_c, u_p, threshold_1=0.5, threshold_2=2.0,
    )
    assert r["status"] == npc.STATUS_INSUFFICIENT


def test_status_conflict_si_unidades_tienen_conflict():
    d = _delta([{"match_status": ds.STATUS_BOTH, "delta_shares": 20.0}])
    u_c = _units([
        {"observed_security_key": "cusip:C1",
         "security_resolution_status": "CONFLICT",
         "canonical_security": None},
    ])
    r = npc.compute_nipc_and_coverage(d, u_c, pd.DataFrame())
    assert r["status"] == npc.STATUS_CONFLICT


def test_status_unresolved_sin_unidades():
    r = npc.compute_nipc_and_coverage(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert r["status"] == npc.STATUS_UNRESOLVED


def test_compute_nipc_and_coverage_devuelve_todas_las_claves():
    d = _delta([{"match_status": ds.STATUS_BOTH, "delta_shares": 20.0}])
    u_c = _units([{"observed_security_key": "cusip:C1"}])
    u_p = _units([{"observed_security_key": "cusip:C1"}])
    r = npc.compute_nipc_and_coverage(d, u_c, u_p)
    expected = {
        "nipc_total", "nipc_sole", "nipc_dfnd", "nipc_otr",
        "n_both", "n_new", "n_exit", "n_unresolved_identity",
        "n_delta_observable",
        "coverage_previous", "coverage_current",
        "paired_security_coverage", "paired_weighted_share_coverage",
        "unmapped_weight_previous", "unmapped_weight_current",
        "status",
    }
    assert expected.issubset(set(r.keys()))


# ---- determinismo ----

def test_nipc_total_available_true_con_datos():
    """P31: nipc_total_available=True cuando hay al menos un delta observable."""
    d = _delta([{"match_status": ds.STATUS_BOTH, "delta_shares": 10.0}])
    r = npc.compute_nipc(d)
    assert r["nipc_total_available"] is True
    assert r["n_delta_observable"] == 1


def test_nipc_total_available_false_sin_datos():
    """P31: nipc_total_available=False cuando no hay observables."""
    r_empty = npc.compute_nipc(pd.DataFrame())
    assert r_empty["nipc_total_available"] is False
    assert r_empty["nipc_total"] == 0.0

    d_unres = _delta([
        {"match_status": ds.STATUS_UNRESOLVED_IDENTITY, "delta_shares": None}
    ])
    r_unres = npc.compute_nipc(d_unres)
    assert r_unres["nipc_total_available"] is False
    assert r_unres["nipc_total"] == 0.0


def test_sin_datetime_now():
    import ast
    import inspect
    src = inspect.getsource(npc)
    tree = ast.parse(src)
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("now", "today"):
            forbidden.append(node.attr)
    assert forbidden == [], f"Llamadas prohibidas: {forbidden}"


# ---- integracion end-to-end con delta_shares ----

def test_end_to_end_units_delta_nipc():
    """Q4->Q1 con dos managers y tres CUSIPs. NIPC observable."""
    info_q4 = pd.DataFrame([
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 1, "CUSIP": "C1",
         "SSHPRNAMTTYPE": "SH", "PUTCALL": None, "SSHPRNAMT": 100.0,
         "INVESTMENTDISCRETION": "SOLE"},
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 2, "CUSIP": "C2",
         "SSHPRNAMTTYPE": "SH", "PUTCALL": None, "SSHPRNAMT": 200.0,
         "INVESTMENTDISCRETION": "DFND"},
    ])
    info_q1 = pd.DataFrame([
        {"ACCESSION_NUMBER": "B1", "INFOTABLE_SK": 1, "CUSIP": "C1",
         "SSHPRNAMTTYPE": "SH", "PUTCALL": None, "SSHPRNAMT": 130.0,
         "INVESTMENTDISCRETION": "SOLE"},
        {"ACCESSION_NUMBER": "B1", "INFOTABLE_SK": 2, "CUSIP": "C2",
         "SSHPRNAMTTYPE": "SH", "PUTCALL": None, "SSHPRNAMT": 150.0,
         "INVESTMENTDISCRETION": "DFND"},
    ])
    sub_q4 = pd.DataFrame([("A1", "FM1")], columns=["ACCESSION_NUMBER", "CIK"])
    sub_q1 = pd.DataFrame([("B1", "FM1")], columns=["ACCESSION_NUMBER", "CIK"])

    idr = {
        "C1": {"observed_security_key": "cusip:C1",
               "security_resolution_status": "CANONICAL",
               "canonical_security_kind": "CANONICAL_EQUIVALENCE",
               "canonical_security": "equity:C1"},
        "C2": {"observed_security_key": "cusip:C2",
               "security_resolution_status": "CANONICAL",
               "canonical_security_kind": "CANONICAL_EQUIVALENCE",
               "canonical_security": "equity:C2"},
    }
    u_q4 = ds.compute_reported_position_units(
        info_q4, sub_q4, report_period="2025-12-31", identity_results=idr)
    u_q1 = ds.compute_reported_position_units(
        info_q1, sub_q1, report_period="2026-03-31", identity_results=idr)
    delta = ds.compute_delta_shares(u_q1, u_q4)
    result = npc.compute_nipc_and_coverage(delta, u_q1, u_q4)

    # NIPC total = (130-100) + (150-200) = +30 - 50 = -20
    assert result["nipc_total"] == -20.0
    # SOLE delta = +30, DFND delta = -50
    assert result["nipc_sole"] == 30.0
    assert result["nipc_dfnd"] == -50.0
    assert result["n_both"] == 2
    assert result["paired_security_coverage"] == 1.0
    assert result["status"] == npc.STATUS_INSUFFICIENT  # sin thresholds

# ---- P38 + Q5 ----

def test_p38_denominador_interseccion_no_union():
    """P38: dos securities solo en Q4 / solo en Q1 no entran al denominador."""
    u_c = _units([{"observed_security_key": "cusip:A"}])
    u_p = _units([{"observed_security_key": "cusip:B"}])
    c = npc.compute_coverage_pairwise(u_c, u_p)
    # TARGET_PAIRWISE = {} -> 0.0, UNAVAILABLE
    assert c["paired_security_coverage"] == 0.0
    assert c["paired_weighted_share_coverage"] is None
    assert c["coverage_status"] == "UNAVAILABLE"


def test_p38_denominador_ambos_presentes():
    """P38: cuando A esta en ambos, A es TARGET_PAIRWISE."""
    u_c = _units([
        {"observed_security_key": "cusip:A", "sshprnamt_total": 100.0},
        {"observed_security_key": "cusip:B", "sshprnamt_total": 200.0},
    ])
    u_p = _units([
        {"observed_security_key": "cusip:A", "sshprnamt_total": 80.0},
    ])
    c = npc.compute_coverage_pairwise(u_c, u_p)
    # TARGET_PAIRWISE = {A}, both_mapped = {A} -> paired_security_coverage = 1.0
    assert c["paired_security_coverage"] == 1.0
    # w(A) = max(100, 80) = 100; numer=100, denom=100 -> 1.0
    assert c["paired_weighted_share_coverage"] == 1.0
    assert c["coverage_status"] == "VALID"


def test_q5_coverage_status_unavailable_cuando_vacio():
    """Q5: cobertura no medible -> None + UNAVAILABLE, no 0.0."""
    c = npc.compute_coverage_pairwise(pd.DataFrame(), pd.DataFrame())
    assert c["paired_weighted_share_coverage"] is None
    assert c["coverage_status"] == "UNAVAILABLE"

