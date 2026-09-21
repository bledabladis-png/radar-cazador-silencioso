"""Tests B3 - timestamps (derive + enrich)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.institutional_accumulation import timestamps as ts


def _snapshot():
    sub = pd.DataFrame([
        {
            "ACCESSION_NUMBER": "acc-A",
            "FILING_DATE": pd.Timestamp("2026-02-15"),
            "PERIODOFREPORT": pd.Timestamp("2025-12-31"),
        },
        {
            "ACCESSION_NUMBER": "acc-B",
            "FILING_DATE": pd.Timestamp("2026-05-10"),
            "PERIODOFREPORT": pd.Timestamp("2026-03-31"),
        },
    ])
    return {"SUBMISSION": sub}

def test_derive_timestamps_basico():
    out = ts.derive_timestamps(_snapshot(), "acc-A")
    assert out == {
        "period_end": "2025-12-31",
        "filing_date": "2026-02-15",
        "knowledge_date": "2026-02-15",
    }


def test_derive_timestamps_opcion_a():
    """Opcion A: knowledge_date == filing_date."""
    out = ts.derive_timestamps(_snapshot(), "acc-B")
    assert out["knowledge_date"] == out["filing_date"]
    assert out["filing_date"] == "2026-05-10"
    assert out["period_end"] == "2026-03-31"


def test_derive_timestamps_accession_inexistente():
    out = ts.derive_timestamps(_snapshot(), "acc-NO")
    assert out == {"period_end": None, "filing_date": None, "knowledge_date": None}


def test_derive_timestamps_snapshot_vacio():
    out = ts.derive_timestamps({}, "acc-A")
    assert out == {"period_end": None, "filing_date": None, "knowledge_date": None}


def test_derive_timestamps_submission_vacio():
    out = ts.derive_timestamps({"SUBMISSION": pd.DataFrame()}, "acc-A")
    assert out == {"period_end": None, "filing_date": None, "knowledge_date": None}

def test_enrich_anade_3_columnas():
    positions = pd.DataFrame([
        {"ACCESSION_NUMBER": "acc-A", "CUSIP": "111"},
        {"ACCESSION_NUMBER": "acc-B", "CUSIP": "222"},
    ])
    out = ts.enrich_positions_with_timestamps(positions, _snapshot())
    assert "period_end" in out.columns
    assert "filing_date" in out.columns
    assert "knowledge_date" in out.columns


def test_enrich_valores_correctos():
    positions = pd.DataFrame([
        {"ACCESSION_NUMBER": "acc-A", "CUSIP": "111"},
        {"ACCESSION_NUMBER": "acc-B", "CUSIP": "222"},
    ])
    out = ts.enrich_positions_with_timestamps(positions, _snapshot())
    row_a = out[out["ACCESSION_NUMBER"] == "acc-A"].iloc[0]
    row_b = out[out["ACCESSION_NUMBER"] == "acc-B"].iloc[0]
    assert row_a["period_end"] == "2025-12-31"
    assert row_a["filing_date"] == "2026-02-15"
    assert row_a["knowledge_date"] == "2026-02-15"
    assert row_b["period_end"] == "2026-03-31"
    assert row_b["filing_date"] == "2026-05-10"


def test_enrich_accession_desconocido_queda_none():
    positions = pd.DataFrame([{"ACCESSION_NUMBER": "acc-NO", "CUSIP": "999"}])
    out = ts.enrich_positions_with_timestamps(positions, _snapshot())
    r = out.iloc[0]
    assert r["period_end"] is None
    assert r["filing_date"] is None
    assert r["knowledge_date"] is None

def test_enrich_no_muta_input():
    positions = pd.DataFrame([{"ACCESSION_NUMBER": "acc-A", "CUSIP": "111"}])
    _ = ts.enrich_positions_with_timestamps(positions, _snapshot())
    assert "period_end" not in positions.columns


def test_enrich_columna_ausente_lanza():
    positions = pd.DataFrame([{"CUSIP": "111"}])
    with pytest.raises(ValueError):
        ts.enrich_positions_with_timestamps(positions, _snapshot())


def test_enrich_idempotente():
    positions = pd.DataFrame([{"ACCESSION_NUMBER": "acc-A", "CUSIP": "111"}])
    out1 = ts.enrich_positions_with_timestamps(positions, _snapshot())
    out2 = ts.enrich_positions_with_timestamps(out1, _snapshot())
    pd.testing.assert_frame_equal(out1, out2)

# --- provenance helpers (commit 3b) ---


def test_build_provenance_vacio():
    assert ts.build_provenance() == {}


def test_build_provenance_solo_accession():
    out = ts.build_provenance(accession="acc-A")
    assert out == {ts.PROVENANCE_KEY_EFFECTIVE_FILING_ACCESSION: "acc-A"}


def test_build_provenance_completo():
    out = ts.build_provenance(
        accession="acc-A",
        knowledge_date_status=ts.KD_STATUS_DERIVED_FROM_FILING_DATE,
    )
    assert out[ts.PROVENANCE_KEY_EFFECTIVE_FILING_ACCESSION] == "acc-A"
    assert (
        out[ts.PROVENANCE_KEY_KNOWLEDGE_DATE_STATUS]
        == ts.KD_STATUS_DERIVED_FROM_FILING_DATE
    )


def test_build_provenance_status_invalido_lanza():
    with pytest.raises(ValueError):
        ts.build_provenance(
            accession="acc-A",
            knowledge_date_status="NO_EXISTE",
        )


def test_kd_status_constantes():
    assert ts.KD_STATUS_DERIVED_FROM_FILING_DATE == "DERIVED_FROM_FILING_DATE"
    assert ts.KD_STATUS_NOT_AVAILABLE == "NOT_AVAILABLE"
    assert ts.ALL_KD_STATUSES == (
        "DERIVED_FROM_FILING_DATE",
        "NOT_AVAILABLE",
    )
