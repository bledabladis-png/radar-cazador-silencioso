# -*- coding: utf-8 -*-
"""Tests de sec_13f.identity.security_identity. Sin red."""
import pandas as pd
import pytest

from src.institutional_accumulation.sec_13f.identity import security_identity as si


# ---- helpers ----

def _mk_equivalence(rows, identity_type="TICKER"):
    """rows: list of (CUSIP_A, canonical, valid_from, valid_to, source, reason, verified_by).

    P60 / F2.4: identity_type es obligatorio. Default TICKER para
    retrocompatibilidad de los tests que no lo especifican.
    """
    df = pd.DataFrame(rows, columns=[
        "CUSIP_A", "canonical_security", "valid_from", "valid_to",
        "source", "reason", "verified_by",
    ])
    df["source_document"] = None
    df["identity_type"] = identity_type
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
    assert si._normalize_canonical("AAPL", "TICKER") == "equity:AAPL"
    assert si._normalize_canonical("equity:AAPL", "TICKER") == "equity:AAPL"
    assert si._normalize_canonical("figi:BBG000B9XRY4", "TICKER") == "figi:BBG000B9XRY4"
    assert si._normalize_canonical("", "TICKER") is None
    assert si._normalize_canonical(None, "TICKER") is None


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
        "CUSIP_A,canonical_security,valid_from,valid_to,source,reason,verified_by,source_document,identity_type\n"
        "X,Y,2026-12-31,2026-01-01,SEC,r,manual,,TICKER\n", encoding="utf-8")
    with pytest.raises(ValueError, match="valid_from > valid_to"):
        si.load_cusip_equivalence(p)


def test_load_cusip_equivalence_verified_by_invalido(tmp_path):
    p = tmp_path / "eq.csv"
    p.write_text(
        "CUSIP_A,canonical_security,valid_from,valid_to,source,reason,verified_by,source_document,identity_type\n"
        "X,Y,2026-01-01,,SEC,r,inventado,,TICKER\n", encoding="utf-8")
    with pytest.raises(ValueError, match="verified_by invalido"):
        si.load_cusip_equivalence(p)


def test_load_cusip_equivalence_source_prohibido(tmp_path):
    p = tmp_path / "eq.csv"
    p.write_text(
        "CUSIP_A,canonical_security,valid_from,valid_to,source,reason,verified_by,source_document,identity_type\n"
        "X,Y,2026-01-01,,manual,r,manual,,TICKER\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source no verificable"):
        si.load_cusip_equivalence(p)


def test_load_cusip_equivalence_solape(tmp_path):
    p = tmp_path / "eq.csv"
    p.write_text(
        "CUSIP_A,canonical_security,valid_from,valid_to,source,reason,verified_by,source_document,identity_type\n"
        "X,Y,2024-01-01,2025-12-31,SEC,r,manual,,TICKER\n"
        "X,Z,2025-06-01,2026-12-31,SEC,r,manual,,TICKER\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Solape"):
        si.load_cusip_equivalence(p)


def test_load_cusip_equivalence_ok(tmp_path):
    p = tmp_path / "eq.csv"
    p.write_text(
        "CUSIP_A,canonical_security,valid_from,valid_to,source,reason,verified_by,source_document,identity_type\n"
        "X,equity:ABC,2024-01-01,2026-12-31,SEC,r,manual,,TICKER\n", encoding="utf-8")
    df = si.load_cusip_equivalence(p)
    assert len(df) == 1
    assert df.iloc[0]["CUSIP_A"] == "X"


# ---- resolve_security_identity ----

def test_resolve_cusip_vacio_unresolved():
    r = si.resolve_security_identity("", "2026-03-31")
    assert r["security_resolution_status"] == si.STATUS_UNRESOLVED
    assert r["canonical_security"] is None
    assert r["canonical_security_kind"] == si.KIND_UNRESOLVED


def test_resolve_sin_fuentes_explicitas_observed_only():
    r = si.resolve_security_identity(
        "037833100", "2026-03-31",
        equivalence_df=pd.DataFrame(),
        crosswalk_internal_df=pd.DataFrame(),
    )
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
        ("A1", "AAPL", None, None, "cusip_ticker_exceptions"),
        ("A1", "MSFT", None, None, "etf_holdings"),
    ])
    r = si.resolve_security_identity(
        "A1", "2026-03-31", crosswalk_internal_df=cw,
    )
    assert r["security_resolution_status"] == si.STATUS_CONFLICT
    assert r["canonical_security"] is None
    # P61 / D1: candidates = [(ticker, source, valid_from, valid_to), ...]
    candidates = {(c[0], c[1]) for c in r["evidence"]["candidates"]}
    assert candidates == {
        ("AAPL", "cusip_ticker_exceptions"),
        ("MSFT", "etf_holdings"),
    }


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


# ---- P60: identity_type obligatorio ----

def test_p60_normalize_canonical_ticker():
    assert si._normalize_canonical("MSFT", "TICKER") == "equity:MSFT"


def test_p60_normalize_canonical_figi():
    assert si._normalize_canonical("BBG001S69V32", "FIGI") == "figi:BBG001S69V32"


def test_p60_normalize_canonical_cusip_no_produce():
    assert si._normalize_canonical("037833100", "CUSIP") is None


def test_p60_normalize_canonical_isin_no_produce():
    assert si._normalize_canonical("US0378331005", "ISIN") is None


def test_p60_identity_type_obligatorio():
    import pytest
    with pytest.raises(ValueError, match="obligatorio"):
        si._normalize_canonical("MSFT", None)


def test_p60_identity_type_invalido():
    import pytest
    with pytest.raises(ValueError, match="invalido"):
        si._normalize_canonical("MSFT", "TICKER_US")

# ---- P61: operational_mapping_status en el resolver ----

def test_p61_equivalence_da_verified():
    eq = _mk_equivalence([
        ("A1", "MSFT", "2024-01-01", "2026-12-31", "SEC", "r", "manual"),
    ])
    r = si.resolve_security_identity("A1", "2026-03-31", equivalence_df=eq)
    assert r["operational_mapping_status"] == "VERIFIED"


def test_p61_crosswalk_da_temporal_unverified():
    """P61 / F2.4: etf_holdings (sin vigencia) -> TEMPORAL_UNVERIFIED."""
    cw = _mk_crosswalk([
        ("A1", "AAPL", None, None, "etf_holdings"),
    ])
    r = si.resolve_security_identity(
        "A1", "2026-03-31", crosswalk_internal_df=cw,
    )
    assert r["operational_mapping_status"] == "TEMPORAL_UNVERIFIED"


def test_p61_sin_resolucion_da_unresolved():
    r = si.resolve_security_identity(
        "A1", "2026-03-31",
        equivalence_df=pd.DataFrame(),
        crosswalk_internal_df=pd.DataFrame(),
    )
    assert r["operational_mapping_status"] == "UNRESOLVED"


def test_p61_figi_lookup_da_temporal_unverified():
    def fake_figi(cusip, period):
        return "BBG001S69V32"
    r = si.resolve_security_identity(
        "A1", "2026-03-31", figi_lookup=fake_figi,
    )
    assert r["operational_mapping_status"] == "TEMPORAL_UNVERIFIED"
    assert r["canonical_security"] == "figi:BBG001S69V32"

# ---- Q8: identity_type en cusip_equivalence.csv ----

def test_p60_csv_sin_identity_type_error(tmp_path):
    """P60 / F2.4: sin columna identity_type, la fuente es invalida."""
    import pandas as pd
    p = tmp_path / "eq.csv"
    pd.DataFrame([
        {"CUSIP_A": "A1", "canonical_security": "MSFT",
         "valid_from": "2024-01-01", "valid_to": "2026-12-31",
         "source": "SEC", "reason": "r", "verified_by": "manual",
         "source_document": None},
    ]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="sin columna identity_type"):
        si.load_cusip_equivalence(p)


def test_q8_identity_type_figi(tmp_path):
    """Con identity_type=FIGI, kind y canonical correctos."""
    import pandas as pd
    p = tmp_path / "eq.csv"
    pd.DataFrame([
        {"CUSIP_A": "A1", "canonical_security": "BBG001S69V32",
         "valid_from": "2024-01-01", "valid_to": "2026-12-31",
         "source": "SEC", "reason": "r", "verified_by": "manual",
         "source_document": None, "identity_type": "FIGI"},
    ]).to_csv(p, index=False)
    eq = si.load_cusip_equivalence(p)
    r = si.resolve_security_identity("A1", "2026-03-31", equivalence_df=eq)
    assert r["security_resolution_status"] == si.STATUS_CANONICAL
    assert r["canonical_security"] == "figi:BBG001S69V32"
    assert r["canonical_security_kind"] == si.KIND_CANONICAL_FIGI



# --- S-03c (2026-09-29): invariante CANONICAL != None en crosswalk ---

def test_s03c_crosswalk_ticker_nan_string_degrada_a_observed_only():
    """Bug: ticker literal 'nan' producia CANONICAL con canonical_security=None."""
    cw = _mk_crosswalk([
        ("A1", "nan", None, None, "etf_holdings"),
    ])
    r = si.resolve_security_identity("A1", "2026-03-31",
                                    crosswalk_internal_df=cw)
    assert r["security_resolution_status"] == si.STATUS_OBSERVED_ONLY
    assert r["canonical_security"] is None
    assert r["canonical_security_kind"] == si.KIND_OBSERVED_CUSIP_ONLY


def test_s03c_crosswalk_ticker_none_string_degrada_a_observed_only():
    cw = _mk_crosswalk([
        ("A1", "none", None, None, "etf_holdings"),
    ])
    r = si.resolve_security_identity("A1", "2026-03-31",
                                    crosswalk_internal_df=cw)
    assert r["security_resolution_status"] == si.STATUS_OBSERVED_ONLY
    assert r["canonical_security"] is None


def test_s03c_crosswalk_ticker_null_string_degrada_a_observed_only():
    cw = _mk_crosswalk([
        ("A1", "null", None, None, "etf_holdings"),
    ])
    r = si.resolve_security_identity("A1", "2026-03-31",
                                    crosswalk_internal_df=cw)
    assert r["security_resolution_status"] == si.STATUS_OBSERVED_ONLY
    assert r["canonical_security"] is None


def test_s03c_crosswalk_ticker_valido_mantiene_canonical():
    """Control positivo: ticker valido sigue produciendo CANONICAL."""
    cw = _mk_crosswalk([
        ("A1", "AAPL", None, None, "etf_holdings"),
    ])
    r = si.resolve_security_identity("A1", "2026-03-31",
                                    crosswalk_internal_df=cw)
    assert r["security_resolution_status"] == si.STATUS_CANONICAL
    assert r["canonical_security"] == "equity:AAPL"


def test_s03c_invariante_canonical_implica_security_no_none():
    """Bateria: para N tickers problematicos, CANONICAL implica
    canonical_security != None."""
    tickers = ["nan", "none", "null", "-", "", "  "]
    for tk in tickers:
        cw = _mk_crosswalk([
            ("A1", tk, None, None, "etf_holdings"),
        ])
        r = si.resolve_security_identity("A1", "2026-03-31",
                                        crosswalk_internal_df=cw)
        if r["security_resolution_status"] == si.STATUS_CANONICAL:
            assert r["canonical_security"] is not None, (
                f"invariante violada para ticker={tk!r}")

# ---- Reverse split HON 2026-06-29 (fix 2026-11-05) ----
# Bug: 507 filings Q2 reportaban CUSIP viejo 438516106 (expirado en crosswalk
# 2026-03-31) y caian a OBSERVED_ONLY -> UNRESOLVED_IDENTITY -> EXIT artificial.
# Fix: cusip_equivalence.csv con 2 entries que mapean ambos CUSIPs a equity:HON.

def _mk_hon_equivalence():
    """Equivalence del fix HON 2026-06-29."""
    return _mk_equivalence([
        ("438516106", "HON", "2025-12-31", "2026-09-30", "SEC-EDGAR", "pre-split", "SEC"),
        ("438516205", "HON", "2026-06-29", None, "SEC-EDGAR", "post-split", "SEC"),
    ])


def _mk_hon_crosswalk():
    """Crosswalk con entry expirada del CUSIP viejo."""
    return _mk_crosswalk([
        ("438516106", "HON", "2025-12-31", "2026-03-31", "cusip_ticker_exceptions"),
        ("43849R105", "HONA", None, None, "etf_holdings"),
    ])


def test_hon_reverse_split_cusip_pre_resuelve_en_q1():
    """Q1: CUSIP viejo 438516106 resuelve a HON (dentro de ventana equivalence)."""
    eq = _mk_hon_equivalence()
    cw = _mk_hon_crosswalk()
    r = si.resolve_security_identity("438516106", "2026-03-31",
                                     equivalence_df=eq, crosswalk_internal_df=cw)
    assert r["canonical_security"] == "equity:HON"
    assert r["security_resolution_status"] == si.STATUS_CANONICAL


def test_hon_reverse_split_cusip_pre_resuelve_en_q2():
    """Q2: CUSIP viejo 438516106 sigue resolviendo a HON (ventana ampliada)."""
    eq = _mk_hon_equivalence()
    cw = _mk_hon_crosswalk()
    r = si.resolve_security_identity("438516106", "2026-06-30",
                                     equivalence_df=eq, crosswalk_internal_df=cw)
    assert r["canonical_security"] == "equity:HON"
    assert r["security_resolution_status"] == si.STATUS_CANONICAL


def test_hon_reverse_split_cusip_pre_fuera_de_ventana_no_resuelve():
    """Post 2026-09-30: CUSIP viejo NO resuelve (extinto)."""
    eq = _mk_hon_equivalence()
    cw = _mk_hon_crosswalk()
    r = si.resolve_security_identity("438516106", "2026-10-01",
                                     equivalence_df=eq, crosswalk_internal_df=cw)
    assert r["canonical_security"] is None
    assert r["security_resolution_status"] == si.STATUS_OBSERVED_ONLY


def test_hon_reverse_split_cusip_post_resuelve_en_q2():
    """Q2: CUSIP nuevo 438516205 resuelve a HON (post-split)."""
    eq = _mk_hon_equivalence()
    cw = _mk_hon_crosswalk()
    r = si.resolve_security_identity("438516205", "2026-06-30",
                                     equivalence_df=eq, crosswalk_internal_df=cw)
    assert r["canonical_security"] == "equity:HON"
    assert r["security_resolution_status"] == si.STATUS_CANONICAL


def test_hon_reverse_split_ambos_cusips_mismo_canonical():
    """Invariante: pre y post split convergen a equity:HON."""
    eq = _mk_hon_equivalence()
    cw = _mk_hon_crosswalk()
    r_pre = si.resolve_security_identity("438516106", "2026-06-30",
                                         equivalence_df=eq, crosswalk_internal_df=cw)
    r_post = si.resolve_security_identity("438516205", "2026-06-30",
                                          equivalence_df=eq, crosswalk_internal_df=cw)
    assert r_pre["canonical_security"] == r_post["canonical_security"] == "equity:HON"


def test_hon_reverse_split_hona_no_se_contamina():
    """HONA (43849R105) sigue resolviendo a HONA, no a HON."""
    eq = _mk_hon_equivalence()
    cw = _mk_hon_crosswalk()
    r = si.resolve_security_identity("43849R105", "2026-06-30",
                                     equivalence_df=eq, crosswalk_internal_df=cw)
    assert r["canonical_security"] == "equity:HONA"
    assert r["canonical_security"] != "equity:HON"



# ---- S30: guard contra resolve_security_identity sin fuentes ----

def test_resolve_sin_kwargs_raise():
    import pytest
    with pytest.raises(ValueError, match="sin fuentes declaradas"):
        si.resolve_security_identity("037833100", "2026-03-31")
