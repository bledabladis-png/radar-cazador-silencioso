# -*- coding: utf-8 -*-
"""Tests FU-002-bymarket en guard_coverage.

Cubre las correcciones del dictamen auditor:
  C-2: no exencion si quality.status == INVALID.
  C-3: _all_markets_valid usa el MISMO threshold del guard.
  Punto 5: UNKNOWN con n>0 (status INVALID) no exime.
  Punto 6: compatibilidad con manifests antiguos (sin by_market).
"""

import json

from scripts.guard_coverage import (
    _all_markets_valid,
    _check_manifest,
)


# ---------- Helpers ----------
def _write_manifest(tmp_path, name, quality):
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    p = d / f"{name}.manifest.json"
    p.write_text(json.dumps({"quality": quality}), encoding="utf-8")
    return str(p)


def _valid_quality(**overrides):
    """quality minimo que pasa el guard. Override campos para casos."""
    q = {
        "last_date": "2026-09-24",
        "expected_session": "2026-09-24",
        "coverage_pct_last": 1.0,
        "status": "VALID",
    }
    q.update(overrides)
    return q


# ---------- _all_markets_valid ----------
def test_all_markets_valid_todos_valid():
    bm = {
        "US_EQUITY": {"n": 262, "status": "VALID", "coverage_at_session": 1.0},
        "XETRA":     {"n": 19,  "status": "VALID", "coverage_at_session": 1.0},
    }
    assert _all_markets_valid(bm, 0.95) is True


def test_all_markets_valid_ignora_n_cero():
    bm = {
        "US_EQUITY": {"n": 262, "status": "VALID", "coverage_at_session": 1.0},
        "LSE":       {"n": 0,   "status": "SKIP",  "coverage_at_session": None},
    }
    assert _all_markets_valid(bm, 0.95) is True


def test_all_markets_valid_falso_si_by_market_vacio():
    assert _all_markets_valid({}, 0.95) is False
    assert _all_markets_valid(None, 0.95) is False


def test_all_markets_valid_falso_si_todos_n_cero():
    bm = {
        "US_EQUITY": {"n": 0, "status": "SKIP", "coverage_at_session": None},
    }
    assert _all_markets_valid(bm, 0.95) is False


def test_all_markets_valid_falso_si_algun_status_no_valid():
    bm = {
        "US_EQUITY": {"n": 262, "status": "VALID", "coverage_at_session": 1.0},
        "XETRA":     {"n": 19,  "status": "INVALID", "coverage_at_session": 0.0},
    }
    assert _all_markets_valid(bm, 0.95) is False


def test_all_markets_valid_falso_si_coverage_bajo_threshold():
    """C-3: cobertura < threshold no exime aunque status sea VALID."""
    bm = {
        "US_EQUITY": {"n": 100, "status": "VALID", "coverage_at_session": 0.96},
    }
    # threshold 0.97 -> 0.96 no pasa
    assert _all_markets_valid(bm, 0.97) is False
    # threshold 0.95 -> 0.96 si pasa
    assert _all_markets_valid(bm, 0.95) is True


def test_all_markets_valid_falso_si_coverage_no_numerico():
    bm = {
        "US_EQUITY": {"n": 100, "status": "VALID", "coverage_at_session": "0.99"},
    }
    assert _all_markets_valid(bm, 0.95) is False


# ---------- _check_manifest: exencion FU-002-bymarket ----------
def test_guard_exime_si_by_market_todos_valid(tmp_path):
    """Caso base: coverage global bajo pero by_market sano -> EXEMPT."""
    quality = _valid_quality(
        coverage_pct_last=0.134,
        status="VALID_WITH_MISSING",
        by_market={
            "US_EQUITY": {"n": 262, "status": "VALID", "coverage_at_session": 1.0},
            "XETRA":     {"n": 19,  "status": "VALID", "coverage_at_session": 1.0},
            "BME":       {"n": 19,  "status": "VALID", "coverage_at_session": 1.0},
            "EURONEXT":  {"n": 13,  "status": "VALID", "coverage_at_session": 1.0},
            "LSE":       {"n": 0,   "status": "SKIP",  "coverage_at_session": None},
        },
    )
    p = _write_manifest(tmp_path, "stock_prices", quality)
    reasons = _check_manifest(p, 0.95)
    assert reasons == []


def test_guard_c2_no_exime_si_global_invalid(tmp_path):
    """C-2: si quality.status == INVALID, no exencion aunque by_market OK."""
    quality = _valid_quality(
        coverage_pct_last=0.134,
        status="INVALID",
        by_market={
            "US_EQUITY": {"n": 262, "status": "VALID", "coverage_at_session": 1.0},
        },
    )
    p = _write_manifest(tmp_path, "stock_prices", quality)
    reasons = _check_manifest(p, 0.95)
    # Debe fallar por status INVALID (sin exencion)
    assert any("status=INVALID" in r for r in reasons)
    # Y tambien fallara por coverage (no exento)
    assert any("coverage_pct_last" in r for r in reasons)


def test_guard_c3_exencion_usa_mismo_threshold(tmp_path):
    """C-3: si el guard corre con 0.97, un mercado con 0.96 no exime."""
    quality = _valid_quality(
        coverage_pct_last=0.50,
        status="VALID_WITH_MISSING",
        by_market={
            "US_EQUITY": {"n": 100, "status": "VALID", "coverage_at_session": 0.96},
        },
    )
    p = _write_manifest(tmp_path, "stock_prices", quality)
    # Con threshold 0.95: exime
    assert _check_manifest(p, 0.95) == []
    # Con threshold 0.97: NO exime
    reasons = _check_manifest(p, 0.97)
    assert any("coverage_pct_last" in r for r in reasons)


def test_guard_no_exime_si_by_market_falta(tmp_path):
    """Manifest antiguo (sin by_market) -> FAIL como antes."""
    quality = _valid_quality(
        coverage_pct_last=0.134,
        status="VALID_WITH_MISSING",
        # sin by_market
    )
    p = _write_manifest(tmp_path, "stock_prices", quality)
    reasons = _check_manifest(p, 0.95)
    assert any("coverage_pct_last" in r for r in reasons)


def test_guard_punto5_unknown_con_n_mayor_cero_no_exime(tmp_path):
    """Punto 5: UNKNOWN con n>0 tiene status INVALID -> no exime."""
    quality = _valid_quality(
        coverage_pct_last=0.134,
        status="VALID_WITH_MISSING",
        by_market={
            "US_EQUITY": {"n": 262, "status": "VALID", "coverage_at_session": 1.0},
            "UNKNOWN":   {"n": 5,   "status": "INVALID", "coverage_at_session": None},
        },
    )
    p = _write_manifest(tmp_path, "stock_prices", quality)
    reasons = _check_manifest(p, 0.95)
    assert any("coverage_pct_last" in r for r in reasons)


def test_guard_no_exime_si_by_market_todos_skip(tmp_path):
    """by_market con todos n=0 -> FAIL (universo vacio, anomalia)."""
    quality = _valid_quality(
        coverage_pct_last=0.134,
        status="VALID_WITH_MISSING",
        by_market={
            "US_EQUITY": {"n": 0, "status": "SKIP", "coverage_at_session": None},
        },
    )
    p = _write_manifest(tmp_path, "stock_prices", quality)
    reasons = _check_manifest(p, 0.95)
    assert any("coverage_pct_last" in r for r in reasons)


def test_guard_no_exime_si_coverage_suficiente(tmp_path):
    """Caso trivial: cobertura OK, sin exencion necesaria."""
    quality = _valid_quality(
        coverage_pct_last=0.99,
        status="VALID",
    )
    p = _write_manifest(tmp_path, "stock_prices", quality)
    assert _check_manifest(p, 0.95) == []