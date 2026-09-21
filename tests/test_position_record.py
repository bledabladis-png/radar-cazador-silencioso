"""Tests B3 - PositionRecord con timestamps (contrato seccion 12)."""
from __future__ import annotations

import pytest

from src.institutional_accumulation.aggregation import coverage


def _base_record(**overrides):
    kwargs = dict(
        period="Q4",
        observed_security_key="cusip:A",
        share_class_figi="FIGI_X",
        canonical_security="equity:X",
        resolution_status="CANONICAL",
        operational_mapping_status="VERIFIED",
        weight=100.0,
    )
    kwargs.update(overrides)
    return coverage.PositionRecord(**kwargs)

def test_record_sin_timestamps_usa_none_por_defecto():
    r = _base_record()
    assert r.period_end is None
    assert r.filing_date is None
    assert r.knowledge_date is None


def test_record_con_timestamps_completos():
    r = _base_record(
        period_end="2026-03-31",
        filing_date="2026-05-15",
        knowledge_date="2026-05-15",
    )
    assert r.period_end == "2026-03-31"
    assert r.filing_date == "2026-05-15"
    assert r.knowledge_date == "2026-05-15"

def test_record_con_solo_period_end():
    r = _base_record(period_end="2026-03-31")
    assert r.period_end == "2026-03-31"
    assert r.filing_date is None
    assert r.knowledge_date is None


def test_record_backward_compatible_kwargs_antiguos():
    """El constructor antiguo (sin timestamps) sigue funcionando."""
    r = coverage.PositionRecord(
        period="Q1",
        observed_security_key="cusip:B",
        share_class_figi="FIGI_Y",
        canonical_security="equity:Y",
        resolution_status="CANONICAL",
        operational_mapping_status="VERIFIED",
        weight=50.0,
        provenance={"source": "test"},
    )
    assert r.period == "Q1"
    assert r.provenance == {"source": "test"}
    assert r.period_end is None

def test_record_es_frozen():
    r = _base_record(period_end="2026-03-31")
    with pytest.raises(Exception):
        r.period_end = "2026-06-30"


def test_record_opcion_a_knowledge_igual_filing():
    """Contrato opcion A (#45 seccion 11): knowledge_date == filing_date."""
    r = _base_record(
        period_end="2026-03-31",
        filing_date="2026-05-15",
        knowledge_date="2026-05-15",
    )
    assert r.knowledge_date == r.filing_date