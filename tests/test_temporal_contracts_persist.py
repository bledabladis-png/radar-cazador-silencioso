# -*- coding: utf-8 -*-
"""Tests de persist_temporal_contracts_result + build_tcr_payload."""
import json
from datetime import date, datetime, timezone

import pytest

from src.temporal_contracts.base import (
    STATUS_BLOCKED,
    STATUS_INSUFFICIENT,
    STATUS_OK,
    STATUS_PENDING,
    STATUS_STALE,
)
from src.temporal_contracts.persist import (
    TCR_SCHEMA_VERSION,
    TCR_STALE_HOURS,
    build_tcr_payload,
    persist_temporal_contracts_result,
)


NOW = datetime(2026, 11, 10, 22, 0, 0, tzinfo=timezone.utc)


def _meta(by_contract, run_id="20261110_220000",
          reference_date=datetime(2026, 11, 10, 23, 0, 0),
          global_last_date=date(2026, 11, 7)):
    return {
        "by_contract": by_contract,
        "global_last_date": global_last_date,
        "reference_date": reference_date,
        "run_id": run_id,
    }


def _entry(status, lag=0, coverage=1.0):
    return {
        "effective_date": date(2026, 11, 7),
        "expected_date": date(2026, 11, 7),
        "lag_days": lag,
        "coverage": coverage,
        "status": status,
    }


def test_schema_version_es_1():
    assert TCR_SCHEMA_VERSION == 1


def test_stale_hours_es_48():
    assert TCR_STALE_HOURS == 48


def test_payload_vacio_contracts_total_cero():
    payload = build_tcr_payload({}, NOW)
    assert payload["contracts_total"] == 0
    assert payload["by_contract"] == {}
    assert all(v == 0 for v in payload["summary"].values())


def test_summary_ok():
    payload = build_tcr_payload(_meta({
        "EQUITY_EOD": _entry(STATUS_OK),
        "INDEX_EOD_USA": _entry(STATUS_OK),
        "FRED_MACRO": _entry(STATUS_OK),
    }), NOW)
    assert payload["contracts_total"] == 3
    assert payload["summary"][STATUS_OK] == 3
    assert payload["summary"][STATUS_STALE] == 0


def test_summary_mixto():
    payload = build_tcr_payload(_meta({
        "A": _entry(STATUS_OK),
        "B": _entry(STATUS_STALE),
        "C": _entry(STATUS_PENDING),
        "D": _entry(STATUS_INSUFFICIENT),
        "E": _entry(STATUS_BLOCKED),
    }), NOW)
    assert payload["summary"] == {
        STATUS_OK: 1, STATUS_STALE: 1, STATUS_PENDING: 1,
        STATUS_INSUFFICIENT: 1, STATUS_BLOCKED: 1,
    }


def test_fechas_a_iso():
    payload = build_tcr_payload(_meta({"A": _entry(STATUS_OK)}), NOW)
    e = payload["by_contract"]["A"]
    assert e["effective_date"] == "2026-11-07"
    assert e["expected_date"] == "2026-11-07"


def test_reference_date_sin_hora():
    payload = build_tcr_payload(_meta(
        {"A": _entry(STATUS_OK)},
        reference_date=datetime(2026, 11, 10, 23, 0, 0),
    ), NOW)
    assert payload["reference_date"] == "2026-11-10"


def test_entry_no_dict_se_marca_unknown():
    payload = build_tcr_payload(_meta({"ROTO": "no soy dict"}), NOW)
    assert payload["by_contract"]["ROTO"]["status"] == "UNKNOWN"
    assert payload["by_contract"]["ROTO"]["effective_date"] is None


def test_build_payload_requiere_now_datetime():
    with pytest.raises(TypeError):
        build_tcr_payload(_meta({"A": _entry(STATUS_OK)}), "no-datetime")


def test_persist_noop_si_by_contract_vacio(tmp_path):
    out = tmp_path / "tcr.json"
    assert persist_temporal_contracts_result({}, output_path=out) is None
    assert not out.exists()


def test_persist_escribe(tmp_path):
    out = tmp_path / "tcr.json"
    payload = persist_temporal_contracts_result(
        _meta({"EQUITY_EOD": _entry(STATUS_OK)}), output_path=out, now=NOW
    )
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 1
    assert loaded["contracts_total"] == 1
    assert loaded == payload


def test_persist_idempotente(tmp_path):
    out = tmp_path / "tcr.json"
    meta = _meta({"A": _entry(STATUS_OK)})
    persist_temporal_contracts_result(meta, output_path=out, now=NOW)
    n1 = out.read_text(encoding="utf-8")
    persist_temporal_contracts_result(meta, output_path=out, now=NOW)
    n2 = out.read_text(encoding="utf-8")
    assert json.loads(n1)["by_contract"] == json.loads(n2)["by_contract"]
    assert not (tmp_path / "tcr.json.tmp").exists()


def test_persist_deriva_now_de_reference_date(tmp_path):
    out = tmp_path / "tcr.json"
    meta = _meta(
        {"A": _entry(STATUS_OK)},
        reference_date=datetime(2026, 11, 10, 23, 0, 0, tzinfo=timezone.utc),
    )
    payload = persist_temporal_contracts_result(meta, output_path=out)
    assert payload is not None
    assert payload["generated_at_utc"].startswith("2026-11-10T23:00:00")
    assert payload["generated_at_utc"].endswith("+00:00")


def test_persist_sin_now_ni_reference_date_raise(tmp_path):
    out = tmp_path / "tcr.json"
    meta = {"by_contract": {"A": _entry(STATUS_OK)}, "run_id": "x"}
    with pytest.raises(ValueError):
        persist_temporal_contracts_result(meta, output_path=out)


def test_generated_at_utc_timezone_utc(tmp_path):
    payload = persist_temporal_contracts_result(
        _meta({"A": _entry(STATUS_OK)}),
        output_path=tmp_path / "tcr.json",
        now=datetime(2026, 11, 10, 23, 0, 0, tzinfo=timezone.utc),
    )
    assert "generated_at_utc" in payload
    assert payload["generated_at_utc"].endswith("+00:00")


def test_generated_at_utc_naive_as_utc(tmp_path):
    payload = persist_temporal_contracts_result(
        _meta({"A": _entry(STATUS_OK)}),
        output_path=tmp_path / "tcr.json",
        now=datetime(2026, 11, 10, 23, 0, 0),
    )
    assert payload["generated_at_utc"].endswith("+00:00")


def test_generated_at_utc_astimezone_madrid(tmp_path):
    from zoneinfo import ZoneInfo
    madrid = ZoneInfo("Europe/Madrid")
    payload = persist_temporal_contracts_result(
        _meta({"A": _entry(STATUS_OK)}),
        output_path=tmp_path / "tcr.json",
        now=datetime(2026, 11, 10, 23, 0, 0, tzinfo=madrid),
    )
    assert payload["generated_at_utc"].endswith("+00:00")
    assert payload["generated_at_utc"].startswith("2026-11-10T22:00:00")
