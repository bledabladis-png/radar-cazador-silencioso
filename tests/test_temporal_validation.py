# -*- coding: utf-8 -*-
"""Tests de la validacion contractual compartida (enmienda v1.1).

Fuente: docs/auditoria/daily_run_gate_enmienda_v1_1.md
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.temporal_contracts.validation import (
    validate_artifact_contracts,
    STATUS_OK, STATUS_WARN, STATUS_BLOCKED, STATUS_UNAVAILABLE,
)
from src.temporal_contracts.registry import CONTRACTS_REGISTRY, is_critical


def _manifest(by_market):
    return {"quality": {"by_market": by_market}}


def _mkt(n=100, status="VALID", cov=0.99):
    return {"n": n, "status": status, "coverage_at_session": cov}


# ============================================================
# Estructura del registro
# ============================================================
def test_equity_eod_es_critico():
    assert is_critical("EQUITY_EOD") is True


def test_indices_no_son_criticos():
    for name in ("INDEX_EOD_USA", "INDEX_EOD_EUROPA",
                 "VOLATILITY_INDEX", "FUTURE_SETTLEMENT"):
        assert is_critical(name) is False


def test_todos_los_contratos_tienen_campo_critical():
    for name, entry in CONTRACTS_REGISTRY.items():
        assert "critical" in entry, f"{name} sin campo critical"


# ============================================================
# Casos de manifest
# ============================================================
def test_manifest_no_dict_es_unavailable():
    r = validate_artifact_contracts("no-dict")
    assert r.status == STATUS_UNAVAILABLE


def test_manifest_sin_quality_es_unavailable():
    r = validate_artifact_contracts({})
    assert r.status == STATUS_UNAVAILABLE


def test_by_market_ausente_es_warn():
    """Enmienda v1.1: sin by_market no hay desglose por mercado.
    La validacion global del guard sigue aplicando. WARN, no OK ni
    UNAVAILABLE."""
    r = validate_artifact_contracts({"quality": {}})
    assert r.status == STATUS_WARN


def test_by_market_vacio_es_warn():
    """Mismo criterio: dict vacio equivale a ausente."""
    r = validate_artifact_contracts({"quality": {"by_market": {}}})
    assert r.status == STATUS_WARN


def test_equity_eod_valido_es_ok():
    m = _manifest({"US_EQUITY": _mkt()})
    r = validate_artifact_contracts(m)
    assert r.status == STATUS_OK


def test_equity_eod_invalid_es_blocked():
    m = _manifest({"US_EQUITY": _mkt(status="INVALID")})
    r = validate_artifact_contracts(m)
    assert r.status == STATUS_BLOCKED


def test_equity_eod_coverage_baja_es_blocked():
    """Regla del auditor: cobertura baja bloquea AUNQUE status != INVALID."""
    m = _manifest({"US_EQUITY": _mkt(status="VALID_WITH_MISSING", cov=0.5)})
    r = validate_artifact_contracts(m)
    assert r.status == STATUS_BLOCKED


def test_equity_eod_n_cero_es_blocked():
    m = _manifest({"US_EQUITY": _mkt(n=0, status="SKIP", cov=None)})
    r = validate_artifact_contracts(m)
    assert r.status == STATUS_BLOCKED


def test_equity_eod_cov_ausente_es_unavailable():
    m = _manifest({"US_EQUITY": _mkt(cov=None)})
    r = validate_artifact_contracts(m)
    assert r.status == STATUS_UNAVAILABLE


def test_contrato_critico_ausente_es_unavailable():
    """by_market sin US_EQUITY -> UNAVAILABLE (no se puede evaluar)."""
    m = _manifest({"BME": _mkt()})
    r = validate_artifact_contracts(m)
    assert r.status == STATUS_UNAVAILABLE


def test_mercado_no_critico_invalid_es_warn():
    m = _manifest({
        "US_EQUITY": _mkt(),
        "BME": _mkt(status="INVALID"),
    })
    r = validate_artifact_contracts(m)
    assert r.status == STATUS_WARN


def test_contrato_critico_ok_mas_no_critico_warn():
    m = _manifest({
        "US_EQUITY": _mkt(),
        "EURONEXT": _mkt(status="INVALID"),
    })
    r = validate_artifact_contracts(m)
    assert r.status == STATUS_WARN


def test_blocks_publication():
    r = validate_artifact_contracts(
        _manifest({"US_EQUITY": _mkt(status="INVALID")})
    )
    assert r.blocks_publication() is True


def test_ok_no_bloquea_publicacion():
    r = validate_artifact_contracts(_manifest({"US_EQUITY": _mkt()}))
    assert r.blocks_publication() is False
    assert r.is_ok_or_warn() is True