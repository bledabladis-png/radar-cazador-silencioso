"""Tests B3 - absence.py stub (contrato semantico seccion 12)."""
from __future__ import annotations

import pytest

from src.institutional_accumulation import absence


def test_reporting_status_enum_completo():
    assert absence.ALL_REPORTING_STATUSES == (
        "PRESENT",
        "ZERO_REPORTED",
        "NOT_PRESENT",
    )


def test_reporting_status_constantes():
    assert absence.REPORTING_PRESENT == "PRESENT"
    assert absence.REPORTING_ZERO_REPORTED == "ZERO_REPORTED"
    assert absence.REPORTING_NOT_PRESENT == "NOT_PRESENT"

def test_absence_reason_enum_completo():
    assert absence.ALL_ABSENCE_REASONS == (
        "MISSING",
        "BELOW_REPORTING_THRESHOLD",
        "CONFIDENTIAL",
        "OTHER_MANAGER",
        "UNKNOWN",
    )


def test_absence_reason_default_es_missing():
    assert absence.DEFAULT_ABSENCE_REASON == "MISSING"


def test_sale_evidence_enum_completo():
    assert absence.ALL_SALE_EVIDENCE == ("NONE", "DIRECT")


def test_sale_evidence_default_es_none():
    assert absence.DEFAULT_SALE_EVIDENCE == "NONE"

def test_sold_no_es_miembro_directo():
    """SOLD no existe como constante directa (contrato seccion 12.2)."""
    assert not hasattr(absence, "REASON_SOLD")
    assert not hasattr(absence, "SALE_SOLD")
    assert "SOLD" not in absence.ALL_ABSENCE_REASONS
    assert "SOLD" not in absence.ALL_SALE_EVIDENCE


def test_classify_absence_lanza_notimplemented():
    with pytest.raises(absence.AbsenceClassificationNotImplemented):
        absence.classify_absence()

def test_classify_absence_es_subclase_notimplemented():
    assert issubclass(
        absence.AbsenceClassificationNotImplemented,
        NotImplementedError,
    )


def test_classify_absence_mensaje_menciona_diferido():
    with pytest.raises(NotImplementedError) as exc_info:
        absence.classify_absence()
    assert "diferida" in str(exc_info.value).lower()