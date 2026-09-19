"""Tests P61 temporal_validity."""
import pytest

from src.institutional_accumulation import temporal_validity as tv


def test_status_enum():
    assert tv.STATUS_VERIFIED == "VERIFIED"
    assert tv.STATUS_TEMPORAL_UNVERIFIED == "TEMPORAL_UNVERIFIED"
    assert tv.STATUS_UNRESOLVED == "UNRESOLVED"
    assert tv.STATUS_CONFLICT == "CONFLICT"


def test_source_sin_vigencia_temporal_unverified():
    assert tv.resolve_source_status("etf_holdings", None, None, "2026-03-31") == tv.STATUS_TEMPORAL_UNVERIFIED
    assert tv.resolve_source_status("openfigi", None, None, "2026-03-31") == tv.STATUS_TEMPORAL_UNVERIFIED


def test_source_con_vigencia_cubre_verified():
    assert tv.resolve_source_status("cusip_ticker_exceptions", "2024-01-01", "2026-12-31", "2026-03-31") == tv.STATUS_VERIFIED


def test_source_con_vigencia_no_cubre_unresolved():
    assert tv.resolve_source_status("cusip_ticker_exceptions", "2024-01-01", "2025-12-31", "2026-03-31") == tv.STATUS_UNRESOLVED


def test_source_con_vigencia_valid_to_null():
    assert tv.resolve_source_status("cusip_ticker_exceptions", "2024-01-01", None, "2026-03-31") == tv.STATUS_VERIFIED


def test_source_desconocida_unresolved():
    assert tv.resolve_source_status("otra_fuente", "2024-01-01", "2026-12-31", "2026-03-31") == tv.STATUS_UNRESOLVED


def test_period_invalido_lanza():
    with pytest.raises(ValueError, match="period invalido"):
        tv.resolve_source_status("etf_holdings", None, None, "no-es-fecha")


def test_aggregate_vacio():
    assert tv.aggregate_status([]) == tv.STATUS_UNRESOLVED


def test_aggregate_un_verified():
    assert tv.aggregate_status([("MSFT", tv.STATUS_VERIFIED)]) == tv.STATUS_VERIFIED


def test_aggregate_dos_verified_distintos_conflict():
    assert tv.aggregate_status([
        ("MSFT", tv.STATUS_VERIFIED),
        ("AAPL", tv.STATUS_VERIFIED),
    ]) == tv.STATUS_CONFLICT


def test_aggregate_verified_y_temp_mismo_valor():
    assert tv.aggregate_status([
        ("MSFT", tv.STATUS_VERIFIED),
        ("MSFT", tv.STATUS_TEMPORAL_UNVERIFIED),
    ]) == tv.STATUS_VERIFIED


def test_aggregate_verified_y_temp_distinto_conflict():
    assert tv.aggregate_status([
        ("MSFT", tv.STATUS_VERIFIED),
        ("AAPL", tv.STATUS_TEMPORAL_UNVERIFIED),
    ]) == tv.STATUS_CONFLICT


def test_aggregate_solo_temp_mismo_valor():
    assert tv.aggregate_status([
        ("MSFT", tv.STATUS_TEMPORAL_UNVERIFIED),
        ("MSFT", tv.STATUS_TEMPORAL_UNVERIFIED),
    ]) == tv.STATUS_TEMPORAL_UNVERIFIED


def test_aggregate_solo_temp_distinto_conflict():
    assert tv.aggregate_status([
        ("MSFT", tv.STATUS_TEMPORAL_UNVERIFIED),
        ("AAPL", tv.STATUS_TEMPORAL_UNVERIFIED),
    ]) == tv.STATUS_CONFLICT


def test_aggregate_solo_unresolved():
    assert tv.aggregate_status([
        ("A", tv.STATUS_UNRESOLVED),
        ("B", tv.STATUS_UNRESOLVED),
    ]) == tv.STATUS_UNRESOLVED