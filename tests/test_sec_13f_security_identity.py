# -*- coding: utf-8 -*-
"""Tests de sec_13f.identity.security_identity. Sin red."""
import pandas as pd
import pytest

from src.institutional_accumulation.sec_13f.identity import security_identity as si


# ---- helpers ----

def _mk_equivalence(rows):
    """rows: list of (CUSIP_A, canonical, valid_from, valid_to, source, reason, verified_by)."""
    df = pd.DataFrame(rows, columns=[
        "CUSIP_A", "canonical_security", "valid_from", "valid_to",
        "source", "reason", "verified_by",
    ])
    df["source_document"] = None
    df["valid_from"] = pd.to_datetime(df["valid_from"])
    df["valid_to"] = pd.to_datetime(df["valid_to"], errors="coerce")
    return df


def _mk_crosswalk(rows):
    """rows: list of (CUSIP, ticker, valid_from, valid_to, source)."""
    df = pd.DataFrame(rows, columns=[
        "CUSIP", "ticker", "valid_from", "valid_to", "source",
    ])
    df["valid_from"] = pd.to_datetime(df["valid_from"], errors="coerce")
    df["valid_to"] = pd.to_datetime(df["valid_to"], errors="coerce")
    return df


# ---- constantes ----

def test_estados_definidos():
    assert si.STATUS_CANONICAL == "CANONICAL"
    assert si.STATUS_OBSERVED_ONLY == "OBSERVED_ONLY"
    assert si.STATUS_UNRESOLVED == "UNRESOLVED"
    assert si.STATUS_AMBIGUOUS == "AMBIGUOUS"
    assert si.STATUS_CONFLICT == "CONFLICT"
    assert len(si.ALL_STATUSES) == 5


def test_kinds_definidos():
    assert si.KIND_CANONICAL_FIGI == "CANONICAL_FIGI"
    assert si.KIND_CANONICAL_EQUIVALENCE == "CANONICAL_EQUIVALENCE"
    assert si.KIND_OBSERVED_CUSIP_ONLY == "OBSERVED_CUSIP_ONLY"
    assert si.KIND_UNRESOLVED == "UNRESOLVED"
    assert len(si.ALL_KINDS) == 4


def test_no_ticker_como_canonical_kind():
    """KIND no incluye 'TICKER' ni 'CANONICAL_TICKER'."""
    for k in si.ALL_KINDS:
        assert "TICKER" not in k


def test_observed_security_key_formato():
    assert si.observed_security_key("037833100") == "cusip:037833100"
    assert si.observed_security_key("  037833100  ") == "cusip:037833100"
    assert si.observed_security_key(None) == "cusip:"


def test_normalize_canonical():
    assert si._normalize_canonical("AAPL") == "equity:AAPL"
    assert si._normalize_canonical("equity:AAPL") == "equity:AAPL"
    assert si._normalize_canonical("figi:BBG000B9XRY4") == "figi:BBG000B9XRY4"
    assert si._normalize_canonical("") is None
    assert si._normalize_canonical(None) is None


# ---- load_cusip_equivalence ----

def test_load_cusip_equivalence_no_existe_devuelve_vacio(tmp_path):
    df = si.load_cusip_equivalence(tmp_path / "no_existe.csv")
    assert df.empty
    assert list(df.columns) == list(si.EQUIVALENCE_COLUMNS)


def test_load_cusip_equivalence_columnas_faltantes(tmp_path):
    p = tmp_path / "eq.csv"
    p.write_text("CUSIP_A,canonical_security\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Columnas faltantes"):
        si.load_cusip_equivalence(p)


def test_load_cusip_equivalence_valid_from_mayor(tmp_path):
    p = tmp_path / "eq.csv"
    p.write_text(
        "CUSIP_A,canonical_security,valid_from,valid_to,source,reason,verified_by,source_document\n"
        "X,Y,2026-12-31,2026-01-01,SEC,r,manual,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="valid_from > valid_to"):
        si.load_cusip_equivalence(p)


def test_load_cusip_equivalence_verified_by_invalido(tmp_path):
    p = tmp_path / "eq.csv"
    p.write_text(
        "CUSIP_A,canonical_security,valid_from,valid_to,source,reason,verified_by,source_document\n"
        "X,Y,2026-01-01,,SEC,r,inventado,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="verified_by invalido"):
        si.load_cusip_equivalence(p)


def test_load_cusip_equivalence_source_prohibido(tmp_path):
    p = tmp_path / "eq.csv"
    p.write_text(
        "CUSIP_A,canonical_security,valid_from,valid_to,source,reason,verified_by,source_document\n"
        "X,Y,2026-01-01,,manual,r,manual,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source no verificable"):
        si.load_cusip_equivalence(p)


def test_load_cusip_equivalence_solape(tmp_path):
    p = tmp_path / "eq.csv"
    p.write_text(
        "CUSIP_A,canonical_security,valid_from,valid_to,source,reason,verified_by,source_document\n"
        "X,Y,2024-01-01,2025-12-31,SEC,r,manual,\n"
        "X,Z,2025-06-01,2026-12-31,SEC,r,manual,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Solape"):
        si.load_cusip_equivalence(p)


def test_load_cusip_equivalence_ok(tmp_path):
    p = tmp_path / "eq.csv"
    p.write_text(
        "CUSIP_A,canonical_security,valid_from,valid_to,source,reason,verified_by,source_document\n"
        "X,equity:ABC,2024-01-01,2026-12-31,SEC,r,manual,\n", encoding="utf-8")
    df = si.load_cusip_equivalence(p)
    assert len(df) == 1
    assert df.iloc[0]["CUSIP_A"] == "X"


# ---- resolve_security_identity ----

def test_resolve_cusip_vacio_unresolved():
    r = si.resolve_security_identity("", "2026-03-31")
    assert r["security_resolution_status"] == si.STATUS_UNRESOLVED
    assert r["canonical_security"] is None
    assert r["canonical_security_kind"] == si.KIND_UNRESOLVED


def test_resolve_sin_fuentes_observed_only():
    r = si.resolve_security_identity("037833100", "2026-03-31")
    assert r["security_resolution_status"] == si.STATUS_OBSERVED_ONLY
    assert r["canonical_security"] is None
    assert r["canonical_security_kind"] == si.KIND_OBSERVED_CUSIP_ONLY
    assert r["observed_security_key"] == "cusip:037833100"


def test_resolve_equivalence_unica():
    eq = _mk_equivalence([
        ("A1", "equity:ABC", "2024-01-01", "2026-12-31", "SEC", "corp action", "manual"),
    ])
    r = si.resolve_security_identity("A1", "2026-03-31", equivalence_df=eq)
    assert r["security_resolution_status"] == si.STATUS_CANONICAL
    assert r["canonical_security_kind"] == si.KIND_CANONICAL_EQUIVALENCE
    assert r["canonical_security"] == "equity:ABC"
    assert r["evidence"]["source"] == "equivalence"


def test_resolve_equivalence_periodo_fuera():
    eq = _mk_equivalence([
        ("A1", "equity:ABC", "2024-01-01", "2024-12-31", "SEC", "r", "manual"),
    ])
    r = si.resolve_security_identity("A1", "2026-03-31", equivalence_df=eq)
    assert r["security_resolution_status"] == si.STATUS_OBSERVED_ONLY


def test_resolve_equivalence_ambigua():
    eq = _mk_equivalence([
        ("A1", "equity:X", "2024-01-01", "2026-12-31", "SEC", "r", "manual"),
        ("A1", "equity:Y", "2025-01-01", "2026-12-31", "SEC", "r", "manual"),
    ])
    r = si.resolve_security_identity("A1", "2026-03-31", equivalence_df=eq)
    assert r["security_resolution_status"] == si.STATUS_AMBIGUOUS
    assert r["canonical_security"] is None
    assert "candidates" in r["evidence"]


def test_resolve_equivalence_normaliza_ticker_a_equity():
    eq = _mk_equivalence([
        ("A1", "AAPL", "2024-01-01", "2026-12-31", "SEC", "r", "manual"),
    ])
    r = si.resolve_security_identity("A1", "2026-03-31", equivalence_df=eq)
    assert r["canonical_security"] == "equity:AAPL"


def test_resolve_equivalence_respeta_figi_prefix():
    eq = _mk_equivalence([
        ("A1", "figi:BBG000B9XRY4", "2024-01-01", "2026-12-31", "SEC", "r", "manual"),
    ])
    r = si.resolve_security_identity("A1", "2026-03-31", equivalence_df=eq)
    assert r["canonical_security"] == "figi:BBG000B9XRY4"


def test_resolve_crosswalk_internal():
    cw = _mk_crosswalk([
        ("A1", "AAPL", "2024-01-01", "2026-12-31", "exceptions"),
    ])
    r = si.resolve_security_identity(
        "A1", "2026-03-31", crosswalk_internal_df=cw,
    )
    assert r["security_resolution_status"] == si.STATUS_CANONICAL
    assert r["canonical_security_kind"] == si.KIND_CANONICAL_EQUIVALENCE
    assert r["canonical_security"] == "equity:AAPL"


def test_resolve_crosswalk_sin_vigencia_activo():
    """etf_holdings no tiene vigencia; debe ser activo siempre."""
    cw = _mk_crosswalk([
        ("A1", "AAPL", None, None, "etf_holdings"),
    ])
    r = si.resolve_security_identity(
        "A1", "2030-01-01", crosswalk_internal_df=cw,
    )
    assert r["security_resolution_status"] == si.STATUS_CANONICAL


def test_resolve_crosswalk_conflicto_varios_tickers():
    cw = _mk_crosswalk([
        ("A1", "AAPL", None, None, "exceptions"),
        ("A1", "MSFT", None, None, "etf_holdings"),
    ])
    r = si.resolve_security_identity(
        "A1", "2026-03-31", crosswalk_internal_df=cw,
    )
    assert r["security_resolution_status"] == si.STATUS_CONFLICT
    assert r["canonical_security"] is None
    assert set(r["evidence"]["candidates"]) == {"AAPL", "MSFT"}


def test_resolve_precedencia_equivalence_sobre_crosswalk():
    eq = _mk_equivalence([
        ("A1", "equity:X", "2024-01-01", "2026-12-31", "SEC", "r", "manual"),
    ])
    cw = _mk_crosswalk([
        ("A1", "Y", None, None, "etf_holdings"),
    ])
    r = si.resolve_security_identity(
        "A1", "2026-03-31",
        equivalence_df=eq, crosswalk_internal_df=cw,
    )
    assert r["canonical_security"] == "equity:X"
    assert r["evidence"]["source"] == "equivalence"


def test_resolve_figi_lookup():
    def fake_lookup(cusip, period):
        return "BBG000B9XRY4"
    r = si.resolve_security_identity(
        "A1", "2026-03-31", figi_lookup=fake_lookup,
    )
    assert r["security_resolution_status"] == si.STATUS_CANONICAL
    assert r["canonical_security_kind"] == si.KIND_CANONICAL_FIGI
    assert r["canonical_security"] == "figi:BBG000B9XRY4"


def test_resolve_figi_lookup_falla_se_ignora():
    def fake_lookup(cusip, period):
        raise RuntimeError("boom")
    r = si.resolve_security_identity(
        "A1", "2026-03-31", figi_lookup=fake_lookup,
    )
    assert r["security_resolution_status"] == si.STATUS_OBSERVED_ONLY


def test_resolve_figi_no_se_usa_si_equivalence_resuelve():
    def fake_lookup(cusip, period):
        return "BBG000B9XRY4"
    eq = _mk_equivalence([
        ("A1", "equity:X", "2024-01-01", "2026-12-31", "SEC", "r", "manual"),
    ])
    r = si.resolve_security_identity(
        "A1", "2026-03-31", equivalence_df=eq, figi_lookup=fake_lookup,
    )
    assert r["canonical_security_kind"] == si.KIND_CANONICAL_EQUIVALENCE


def test_resolve_no_muta_inputs():
    eq = _mk_equivalence([
        ("A1", "equity:X", "2024-01-01", "2026-12-31", "SEC", "r", "manual"),
    ])
    orig = eq.copy()
    si.resolve_security_identity("A1", "2026-03-31", equivalence_df=eq)
    pd.testing.assert_frame_equal(eq, orig)


# ---- resolve_batch_identities ----

def test_resolve_batch():
    eq = _mk_equivalence([
        ("A1", "equity:X", "2024-01-01", "2026-12-31", "SEC", "r", "manual"),
    ])
    r = si.resolve_batch_identities(
        ["A1", "B2"], "2026-03-31", equivalence_df=eq,
    )
    assert r["A1"]["security_resolution_status"] == si.STATUS_CANONICAL
    assert r["B2"]["security_resolution_status"] == si.STATUS_OBSERVED_ONLY


# ---- compute_identity_coverage ----

def test_compute_identity_coverage():
    eq = _mk_equivalence([
        ("A1", "equity:X", "2024-01-01", "2026-12-31", "SEC", "r", "manual"),
    ])
    m = si.compute_identity_coverage(
        ["A1", "B2", "C3"], "2026-03-31", equivalence_df=eq,
    )
    assert m["n_total"] == 3
    assert m["n_canonical"] == 1
    assert m["n_observed_only"] == 2
    assert abs(m["pct_canonical"] - 1.0/3.0) < 1e-9


def test_compute_identity_coverage_vacio():
    m = si.compute_identity_coverage([], "2026-03-31")
    assert m["n_total"] == 0
    assert m["pct_canonical"] == 0.0