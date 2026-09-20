# -*- coding: utf-8 -*-
"""Tests de aggregation.delta_shares (dictamen NIPC v1.1 familias 12.1-12.6).

Sin red. Deterministas. Sin datetime.now().
"""
import pandas as pd
import pytest

from src.institutional_accumulation.aggregation import delta_shares as ds


# ---- helpers ----

def _mk_infotable(rows):
    """rows: list of dicts con al menos ACCESSION_NUMBER, INFOTABLE_SK,
    CUSIP, SSHPRNAMTTYPE, PUTCALL, SSHPRNAMT, INVESTMENTDISCRETION."""
    cols = [
        "ACCESSION_NUMBER", "INFOTABLE_SK", "CUSIP", "SSHPRNAMTTYPE",
        "PUTCALL", "SSHPRNAMT", "INVESTMENTDISCRETION",
    ]
    default = {
        "SSHPRNAMTTYPE": "SH",
        "PUTCALL": None,
        "SSHPRNAMT": 100.0,
        "INVESTMENTDISCRETION": "SOLE",
    }
    full = []
    for r in rows:
        full.append({**default, **r})
    return pd.DataFrame(full, columns=cols)


def _mk_submission(rows):
    """rows: list of (ACCESSION_NUMBER, CIK)."""
    return pd.DataFrame(rows, columns=["ACCESSION_NUMBER", "CIK"])


def _identity_entry(cusip, canonical="equity:AAPL", status="CANONICAL",
                    kind="CANONICAL_EQUIVALENCE", op_status="VERIFIED"):
    return {
        "observed_security_key": "cusip:" + cusip,
        "security_resolution_status": status,
        "canonical_security_kind": kind,
        "canonical_security": canonical,
        "operational_mapping_status": op_status,
    }


# ---- constantes ----

def test_match_key_y_estados_definidos():
    assert ds.MATCH_KEY == ("filing_manager_cik", "canonical_security", "discretion_type")
    assert ds.STATUS_BOTH == "BOTH"
    assert ds.STATUS_NEW == "NEW"
    assert ds.STATUS_EXIT == "EXIT"
    assert ds.STATUS_UNRESOLVED_IDENTITY == "UNRESOLVED_IDENTITY"
    assert "report_period" not in ds.MATCH_KEY


def test_discretions_validas():
    assert set(ds.VALID_DISCRETIONS) == {"SOLE", "DFND", "OTR"}


# ---- familia 12.1: reported_position_unit ----

def test_multi_manager_no_multiplica_shares():
    """OTHERMANAGER con varios valores -> SSHPRNAMT entra UNA SOLA VEZ.

    El dictamen exige que la agregacion sea por source line, NO por edges
    OTHERMANAGER. Este test comprueba que con 1 source line y SSHPRNAMT=X,
    el total por unit es X, no N*X.
    """
    info = _mk_infotable([
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 1, "CUSIP": "037833100",
         "SSHPRNAMT": 1000.0, "INVESTMENTDISCRETION": "SOLE"},
    ])
    sub = _mk_submission([("A1", "FM1")])
    units = ds.compute_reported_position_units(
        info, sub, report_period="2026-03-31",
    )
    assert len(units) == 1
    assert float(units.iloc[0]["sshprnamt_total"]) == 1000.0
    assert int(units.iloc[0]["n_source_lines"]) == 1


def test_discretion_separada():
    """SOLE + DFND en mismo filing -> 2 units."""
    info = _mk_infotable([
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 1, "CUSIP": "037833100",
         "SSHPRNAMT": 1000.0, "INVESTMENTDISCRETION": "SOLE"},
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 2, "CUSIP": "037833100",
         "SSHPRNAMT": 500.0, "INVESTMENTDISCRETION": "DFND"},
    ])
    sub = _mk_submission([("A1", "FM1")])
    units = ds.compute_reported_position_units(
        info, sub, report_period="2026-03-31",
    )
    assert len(units) == 2
    by_disc = units.set_index("discretion_type")["sshprnamt_total"].to_dict()
    assert float(by_disc["SOLE"]) == 1000.0
    assert float(by_disc["DFND"]) == 500.0


def test_reported_position_unit_key_estable_entre_periodos():
    """Misma (filing_manager, canonical_security, discretion) entre Q4/Q1."""
    info_q4 = _mk_infotable([
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 1, "CUSIP": "037833100",
         "SSHPRNAMT": 100.0, "INVESTMENTDISCRETION": "SOLE"},
    ])
    info_q1 = _mk_infotable([
        {"ACCESSION_NUMBER": "B1", "INFOTABLE_SK": 1, "CUSIP": "037833100",
         "SSHPRNAMT": 150.0, "INVESTMENTDISCRETION": "SOLE"},
    ])
    sub_q4 = _mk_submission([("A1", "FM1")])
    sub_q1 = _mk_submission([("B1", "FM1")])
    idr = {"037833100": _identity_entry("037833100")}
    u_q4 = ds.compute_reported_position_units(
        info_q4, sub_q4, report_period="2025-12-31", identity_results=idr)
    u_q1 = ds.compute_reported_position_units(
        info_q1, sub_q1, report_period="2026-03-31", identity_results=idr)
    delta = ds.compute_delta_shares(u_q1, u_q4)
    assert len(delta) == 1
    assert delta.iloc[0]["match_status"] == ds.STATUS_BOTH
    assert float(delta.iloc[0]["delta_shares"]) == 50.0


# ---- familia 12.2: filtros canonicos ----

def test_filtro_sh_null_excluye_prn_y_putcall():
    info = _mk_infotable([
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 1, "CUSIP": "C1",
         "SSHPRNAMT": 100.0, "SSHPRNAMTTYPE": "PRN"},  # excluido
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 2, "CUSIP": "C1",
         "SSHPRNAMT": 200.0, "PUTCALL": "Call"},      # excluido
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 3, "CUSIP": "C1",
         "SSHPRNAMT": 300.0, "PUTCALL": "Put"},       # excluido
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 4, "CUSIP": "C1",
         "SSHPRNAMT": 400.0},                          # valido
    ])
    sub = _mk_submission([("A1", "FM1")])
    units = ds.compute_reported_position_units(
        info, sub, report_period="2026-03-31")
    assert len(units) == 1
    assert float(units.iloc[0]["sshprnamt_total"]) == 400.0
    assert int(units.iloc[0]["n_source_lines"]) == 1


def test_discretion_no_excluye_dfnd():
    info = _mk_infotable([
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 1, "CUSIP": "C1",
         "SSHPRNAMT": 100.0, "INVESTMENTDISCRETION": "DFND"},
    ])
    sub = _mk_submission([("A1", "FM1")])
    units = ds.compute_reported_position_units(
        info, sub, report_period="2026-03-31")
    assert len(units) == 1
    assert units.iloc[0]["discretion_type"] == "DFND"


def test_discretion_invalida_se_excluye():
    info = _mk_infotable([
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 1, "CUSIP": "C1",
         "SSHPRNAMT": 100.0, "INVESTMENTDISCRETION": "BOGUS"},
    ])
    sub = _mk_submission([("A1", "FM1")])
    units = ds.compute_reported_position_units(
        info, sub, report_period="2026-03-31")
    assert len(units) == 0


# ---- familia 12.5: determinismo ----

def test_sin_datetime_now():
    """Verifica por AST que no se llama a now()/today().

    NO busca en docstrings (que si mencionan el termino por contrato).
    Solo detecta llamadas reales en el codigo ejecutable.
    """
    import ast
    import inspect
    src = inspect.getsource(ds)
    tree = ast.parse(src)
    forbidden = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("now", "today"):
            forbidden.append(node.attr)
    assert forbidden == [], f"Llamadas prohibidas detectadas: {forbidden}"


def test_reproducible_mismo_input_mismo_output():
    info = _mk_infotable([
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 1, "CUSIP": "C1",
         "SSHPRNAMT": 100.0},
    ])
    sub = _mk_submission([("A1", "FM1")])
    r1 = ds.compute_reported_position_units(
        info, sub, report_period="2026-03-31")
    r2 = ds.compute_reported_position_units(
        info, sub, report_period="2026-03-31")
    pd.testing.assert_frame_equal(r1, r2)


# ---- familia 12.6: Q-ESP-6 ----

def test_cusip_change_same_canonical_security():
    """Q4 CUSIP_A y Q1 CUSIP_B -> mismo canonical_security -> BOTH, delta real.

    Sin esta capa, seria NEW(A) + EXIT(B) falso.
    """
    info_q4 = _mk_infotable([
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 1, "CUSIP": "CUSIP_A",
         "SSHPRNAMT": 100.0, "INVESTMENTDISCRETION": "SOLE"},
    ])
    info_q1 = _mk_infotable([
        {"ACCESSION_NUMBER": "B1", "INFOTABLE_SK": 1, "CUSIP": "CUSIP_B",
         "SSHPRNAMT": 120.0, "INVESTMENTDISCRETION": "SOLE"},
    ])
    sub_q4 = _mk_submission([("A1", "FM1")])
    sub_q1 = _mk_submission([("B1", "FM1")])
    idr = {
        "CUSIP_A": _identity_entry("CUSIP_A", canonical="equity:X"),
        "CUSIP_B": _identity_entry("CUSIP_B", canonical="equity:X"),
    }
    u_q4 = ds.compute_reported_position_units(
        info_q4, sub_q4, report_period="2025-12-31", identity_results=idr)
    u_q1 = ds.compute_reported_position_units(
        info_q1, sub_q1, report_period="2026-03-31", identity_results=idr)
    delta = ds.compute_delta_shares(u_q1, u_q4)
    assert len(delta) == 1
    assert delta.iloc[0]["match_status"] == ds.STATUS_BOTH
    assert float(delta.iloc[0]["delta_shares"]) == 20.0


def test_multi_edge_does_not_duplicate_delta():
    """Varias source lines del mismo canonical -> 1 unit, delta unico."""
    info = _mk_infotable([
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 1, "CUSIP": "C1",
         "SSHPRNAMT": 100.0},
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 2, "CUSIP": "C1",
         "SSHPRNAMT": 50.0},
    ])
    sub = _mk_submission([("A1", "FM1")])
    idr = {"C1": _identity_entry("C1")}
    units = ds.compute_reported_position_units(
        info, sub, report_period="2026-03-31", identity_results=idr)
    assert len(units) == 1
    assert float(units.iloc[0]["sshprnamt_total"]) == 150.0
    assert int(units.iloc[0]["n_source_lines"]) == 2


def test_unresolved_mapping_blocks_pair_match():
    """Security mapeada en Q1 pero no en Q4 -> UNRESOLVED_IDENTITY, no NEW."""
    info_q4 = _mk_infotable([
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 1, "CUSIP": "C_RES",
         "SSHPRNAMT": 100.0},
    ])
    info_q1 = _mk_infotable([
        {"ACCESSION_NUMBER": "B1", "INFOTABLE_SK": 1, "CUSIP": "C_UNRES",
         "SSHPRNAMT": 120.0},
    ])
    sub_q4 = _mk_submission([("A1", "FM1")])
    sub_q1 = _mk_submission([("B1", "FM1")])
    idr = {
        "C_RES": _identity_entry("C_RES", canonical="equity:X"),
        "C_UNRES": {
            "observed_security_key": "cusip:C_UNRES",
            "security_resolution_status": "OBSERVED_ONLY",
            "canonical_security_kind": "OBSERVED_CUSIP_ONLY",
            "canonical_security": None,
        },
    }
    u_q4 = ds.compute_reported_position_units(
        info_q4, sub_q4, report_period="2025-12-31", identity_results=idr)
    u_q1 = ds.compute_reported_position_units(
        info_q1, sub_q1, report_period="2026-03-31", identity_results=idr)
    delta = ds.compute_delta_shares(u_q1, u_q4)
    # u_q4 (C_RES) queda como EXIT, u_q1 (C_UNRES) queda como UNRESOLVED_IDENTITY
    statuses = set(delta["match_status"])
    assert ds.STATUS_UNRESOLVED_IDENTITY in statuses
    # No debe haber NEW falso para C_UNRES
    unres_row = delta[delta["match_status"] == ds.STATUS_UNRESOLVED_IDENTITY]
    assert len(unres_row) == 1
    assert pd.isna(unres_row.iloc[0]["delta_shares"])


def test_new_and_exit_zero_baseline_current():
    """NEW -> previous=0. EXIT -> current=0. Nunca imputar."""
    info_q4 = _mk_infotable([
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 1, "CUSIP": "C_EXIT",
         "SSHPRNAMT": 100.0},
    ])
    info_q1 = _mk_infotable([
        {"ACCESSION_NUMBER": "B1", "INFOTABLE_SK": 1, "CUSIP": "C_NEW",
         "SSHPRNAMT": 120.0},
    ])
    sub_q4 = _mk_submission([("A1", "FM1")])
    sub_q1 = _mk_submission([("B1", "FM1")])
    idr = {
        "C_EXIT": _identity_entry("C_EXIT", canonical="equity:EXIT"),
        "C_NEW": _identity_entry("C_NEW", canonical="equity:NEW"),
    }
    u_q4 = ds.compute_reported_position_units(
        info_q4, sub_q4, report_period="2025-12-31", identity_results=idr)
    u_q1 = ds.compute_reported_position_units(
        info_q1, sub_q1, report_period="2026-03-31", identity_results=idr)
    delta = ds.compute_delta_shares(u_q1, u_q4)
    by_status = delta.set_index("match_status")
    assert ds.STATUS_NEW in by_status.index
    assert ds.STATUS_EXIT in by_status.index
    new_row = by_status.loc[ds.STATUS_NEW]
    exit_row = by_status.loc[ds.STATUS_EXIT]
    assert float(new_row["sshprnamt_previous"]) == 0.0
    assert float(new_row["sshprnamt_current"]) == 120.0
    assert float(exit_row["sshprnamt_current"]) == 0.0
    assert float(exit_row["sshprnamt_previous"]) == 100.0


# ---- integridad adicional ----

def test_sin_columnas_requeridas_lanza():
    info = pd.DataFrame({"X": [1]})
    sub = _mk_submission([("A1", "FM1")])
    with pytest.raises(KeyError):
        ds.compute_reported_position_units(
            info, sub, report_period="2026-03-31")


def test_no_muta_inputs():
    info = _mk_infotable([
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 1, "CUSIP": "C1",
         "SSHPRNAMT": 100.0},
    ])
    sub = _mk_submission([("A1", "FM1")])
    info_orig = info.copy()
    sub_orig = sub.copy()
    ds.compute_reported_position_units(info, sub, report_period="2026-03-31")
    pd.testing.assert_frame_equal(info, info_orig)
    pd.testing.assert_frame_equal(sub, sub_orig)


def test_reported_position_unit_columns_presentes():
    info = _mk_infotable([
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 1, "CUSIP": "C1",
         "SSHPRNAMT": 100.0},
    ])
    sub = _mk_submission([("A1", "FM1")])
    units = ds.compute_reported_position_units(
        info, sub, report_period="2026-03-31")
    for c in ds.UNITS_COLUMNS:
        assert c in units.columns, c


def test_filter_canonical_stats_presentes():
    """P32: _filter_canonical devuelve (df, stats) con 5 claves."""
    info = _mk_infotable([
        {"CUSIP": "037833100", "SSHPRNAMTTYPE": "SH", "PUTCALL": None,
         "SSHPRNAMT": 100.0, "INVESTMENTDISCRETION": "SOLE"},
    ])
    df, stats = ds._filter_canonical(info)
    assert len(df) == 1
    for k in ("n_rows_input", "n_after_sh_putcall_null",
              "n_dropped_not_sh_or_putcall",
              "n_dropped_invalid_sshprnamt", "n_rows_output"):
        assert k in stats, f"falta clave {k}"
    assert stats["n_rows_input"] == 1
    assert stats["n_rows_output"] == 1
    assert stats["n_dropped_invalid_sshprnamt"] == 0


def test_filter_canonical_cuenta_invalid_sshprnamt():
    """P32: SSHPRNAMT no parseable se cuenta en n_dropped_invalid_sshprnamt."""
    info = _mk_infotable([
        {"CUSIP": "037833100", "SSHPRNAMTTYPE": "SH", "PUTCALL": None,
         "SSHPRNAMT": 100.0, "INVESTMENTDISCRETION": "SOLE"},
        {"CUSIP": "594918104", "SSHPRNAMTTYPE": "SH", "PUTCALL": None,
         "SSHPRNAMT": "abc", "INVESTMENTDISCRETION": "SOLE"},  # invalido
        {"CUSIP": "023135106", "SSHPRNAMTTYPE": "SH", "PUTCALL": None,
         "SSHPRNAMT": "XYZ", "INVESTMENTDISCRETION": "SOLE"},  # invalido
    ])
    df, stats = ds._filter_canonical(info)
    assert stats["n_rows_input"] == 3
    assert stats["n_dropped_invalid_sshprnamt"] == 2
    assert stats["n_rows_output"] == 1


def test_compute_reported_position_units_return_stats():
    """P32: kwarg return_stats=True devuelve (DataFrame, stats)."""
    info = _mk_infotable([
        {"CUSIP": "037833100", "SSHPRNAMTTYPE": "SH", "PUTCALL": None,
         "SSHPRNAMT": 100.0, "INVESTMENTDISCRETION": "SOLE"},
    ])
    sub = _mk_submission([("A1", "FM1")])
    out = ds.compute_reported_position_units(
        info, sub, report_period="2026-03-31", return_stats=True,
    )
    assert isinstance(out, tuple) and len(out) == 2
    units, stats = out
    assert len(units) == 1
    assert stats["n_rows_output"] == 1
    assert "n_dropped_invalid_sshprnamt" in stats


def test_delta_shares_both_con_nan_aborta(monkeypatch):
    """P14-BIS: fila BOTH con NaN -> ValueError, no imputar 0 silencioso."""
    import pandas as pd
    from src.institutional_accumulation.aggregation import delta_shares as ds_mod

    cur_agg_fake = pd.DataFrame({
        "filing_manager_cik": ["FM1"],
        "canonical_security": ["equity:AAPL"],
        "discretion_type": ["SOLE"],
        "sshprnamt_current": [float("nan")],
    })
    prev_agg_fake = pd.DataFrame({
        "filing_manager_cik": ["FM1"],
        "canonical_security": ["equity:AAPL"],
        "discretion_type": ["SOLE"],
        "sshprnamt_previous": [100.0],
    })

    def fake_agg(units_df, col):
        return cur_agg_fake if col == "sshprnamt_current" else prev_agg_fake

    monkeypatch.setattr(ds_mod, "_agg_by_match_key", fake_agg)

    units = pd.DataFrame({
        "filing_manager_cik": ["FM1"],
        "observed_security_key": ["cusip:X"],
        "security_resolution_status": ["CANONICAL"],
        "canonical_security_kind": ["CANONICAL_EQUIVALENCE"],
        "canonical_security": ["equity:AAPL"],
        "discretion_type": ["SOLE"],
        "sshprnamt_total": [100.0],
        "report_period": ["2026-03-31"],
        "n_source_lines": [1],
    })

    with pytest.raises(ValueError, match="corrupcion"):
        ds_mod.compute_delta_shares(units, units.copy())


def test_delta_shares_columns_presentes():
    info = _mk_infotable([
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 1, "CUSIP": "C1",
         "SSHPRNAMT": 100.0},
    ])
    sub = _mk_submission([("A1", "FM1")])
    idr = {"C1": _identity_entry("C1")}
    u = ds.compute_reported_position_units(
        info, sub, report_period="2026-03-31", identity_results=idr)
    delta = ds.compute_delta_shares(u, pd.DataFrame(columns=list(ds.UNITS_COLUMNS)))
    for c in ds.DELTA_COLUMNS:
        assert c in delta.columns, c

# ---- P63 / F2.4 (2026-09-20): Missing != Sold ----


def _mk_am_submission(rows):
    """rows: list of (ACCESSION_NUMBER, CIK, PERIODOFREPORT, FILING_DATE, SUBMISSIONTYPE)."""
    return pd.DataFrame(
        rows,
        columns=["ACCESSION_NUMBER", "CIK", "PERIODOFREPORT",
                 "FILING_DATE", "SUBMISSIONTYPE"],
    )


def _mk_am_coverpage(rows):
    """rows: list of (ACCESSION_NUMBER, AMENDMENTNO, AMENDMENTTYPE, ISAMENDMENT)."""
    return pd.DataFrame(
        rows,
        columns=["ACCESSION_NUMBER", "AMENDMENTNO", "AMENDMENTTYPE", "ISAMENDMENT"],
    )


def test_p63_amendment_evita_exit_falso():
    """P63 / F2.4 Regla 6: amendments resueltos ANTES de delta.

    Escenario:
        Q4:        AAPL presente (100)
        Q1 orig:   AAPL ausente (solo MSFT)
        Q1/A:      NEW HOLDINGS -> AAPL anadido (80)
        Q1 efect:  AAPL presente

    Resultado contractual: AAPL en delta debe ser BOTH, NO EXIT.

    Si el sistema usara el filing original (sin el amendment) en lugar
    del snapshot efectivo, produciria EXIT padre -> venta falsa.
    """
    from src.institutional_accumulation.sec_13f.identity.amendments import (
        apply_amendments,
    )

    # Q4 (2025-12-31): base filing X1 con AAPL.
    q4_dfs = {
        "SUBMISSION": _mk_am_submission([
            ("X1", "9999", "2025-12-31", "2026-02-14", "13F-HR"),
        ]),
        "COVERPAGE": _mk_am_coverpage([
            ("X1", None, None, None),
        ]),
        "INFOTABLE": _mk_infotable([
            {"ACCESSION_NUMBER": "X1", "INFOTABLE_SK": "1",
             "CUSIP": "037833100", "SSHPRNAMT": 100.0},
        ]),
    }

    # Q1 (2026-03-31): base A1 sin AAPL + amendment A2 con NEW HOLDINGS.
    q1_dfs = {
        "SUBMISSION": _mk_am_submission([
            ("A1", "9999", "2026-03-31", "2026-05-14", "13F-HR"),
            ("A2", "9999", "2026-03-31", "2026-05-20", "13F-HR/A"),
        ]),
        "COVERPAGE": _mk_am_coverpage([
            ("A1", None, None, None),
            ("A2", 1, "NEW HOLDINGS", "Y"),
        ]),
        "INFOTABLE": _mk_infotable([
            {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": "1",
             "CUSIP": "594918104", "SSHPRNAMT": 50.0},
            {"ACCESSION_NUMBER": "A2", "INFOTABLE_SK": "1",
             "CUSIP": "037833100", "SSHPRNAMT": 80.0},
        ]),
    }

    snap_q4 = apply_amendments(q4_dfs, period="2025-12-31")["canonical_snapshot"]
    snap_q1 = apply_amendments(q1_dfs, period="2026-03-31")["canonical_snapshot"]

    # Verificar que el snapshot efectivo Q1 incluye el amendment A2.
    assert "A2" in snap_q1["SUBMISSION"]["ACCESSION_NUMBER"].tolist(), (
        "el snapshot efectivo Q1 debe incluir el amendment A2"
    )
    assert "037833100" in snap_q1["INFOTABLE"]["CUSIP"].tolist(), (
        "el snapshot efectivo Q1 debe incluir AAPL via amendment"
    )

    # Identidad: ambos CUSIPs resueltos.
    identity = {
        "037833100": _identity_entry("037833100", canonical="equity:AAPL"),
        "594918104": _identity_entry("594918104", canonical="equity:MSFT"),
    }

    units_q4 = ds.compute_reported_position_units(
        snap_q4["INFOTABLE"], snap_q4["SUBMISSION"],
        snap_q4.get("COVERPAGE"),
        report_period="2025-12-31", identity_results=identity,
    )
    units_q1 = ds.compute_reported_position_units(
        snap_q1["INFOTABLE"], snap_q1["SUBMISSION"],
        snap_q1.get("COVERPAGE"),
        report_period="2026-03-31", identity_results=identity,
    )

    delta = ds.compute_delta_shares(units_q1, units_q4)

    aapl = delta[delta["canonical_security"] == "equity:AAPL"]
    assert len(aapl) == 1, f"esperado 1 fila AAPL, obtenido {len(aapl)}"
    assert aapl.iloc[0]["match_status"] == ds.STATUS_BOTH, (
        "P63 Regla 6: AAPL debe ser BOTH (el amendment la anade al "
        "snapshot efectivo). Obtenido: " + str(aapl.iloc[0]["match_status"])
    )
    assert aapl.iloc[0]["delta_shares"] == -20.0, (
        "delta esperado = 80 - 100 = -20"
    )


def test_p63_othermanager_produce_exit_mas_new():
    """P63 / F2.4 Regla 7: 13F admite Other Manager / Combination Report.

    Comportamiento actual documentado:
        Q4: filing_manager_cik=parent, AAPL=100
        Q1: filing_manager_cik=child,  AAPL=100

    Sin resolver relaciones OTHERMANAGER2, el reconciliador produce:
        EXIT en parent (sshprnamt_current=0)
        NEW  en child  (sshprnamt_previous=0)

    Esto NO es un bug: el sistema implementa la reconciliacion C2 fielmente.
    La resolucion de relaciones (distinguir reasignacion intra-grupo de
    venta+compra) queda como capacidad diferida v1 (filer continuity =
    caracterizacion, no threshold).

    F2.4 Regla 7 lo declara explicitamente en el contrato.
    """
    # Q4: parent reporta AAPL.
    u_q4 = ds.compute_reported_position_units(
        _mk_infotable([
            {"ACCESSION_NUMBER": "P1", "INFOTABLE_SK": "1",
             "CUSIP": "037833100", "SSHPRNAMT": 100.0},
        ]),
        _mk_submission([("P1", "1111")]),
        None,
        report_period="2025-12-31",
        identity_results={
            "037833100": _identity_entry("037833100", canonical="equity:AAPL"),
        },
    )

    # Q1: child reporta AAPL (mismo canonical, distinto filing_manager_cik).
    u_q1 = ds.compute_reported_position_units(
        _mk_infotable([
            {"ACCESSION_NUMBER": "C1", "INFOTABLE_SK": "1",
             "CUSIP": "037833100", "SSHPRNAMT": 100.0},
        ]),
        _mk_submission([("C1", "2222")]),
        None,
        report_period="2026-03-31",
        identity_results={
            "037833100": _identity_entry("037833100", canonical="equity:AAPL"),
        },
    )

    delta = ds.compute_delta_shares(u_q1, u_q4)
    assert len(delta) == 2, f"esperado 2 filas (EXIT + NEW), obtenido {len(delta)}"

    statuses = set(delta["match_status"].tolist())
    assert statuses == {ds.STATUS_EXIT, ds.STATUS_NEW}, (
        "comportamiento actual: EXIT en parent + NEW en child. Obtenido: "
        + str(statuses)
    )

    # Verificar que NO se ha fabricado SOLD.
    assert "SOLD" not in ds.ALL_MATCH_STATUSES, (
        "P63: 13F aislado no puede inferir SOLD. SOLD no debe existir como "
        "estado de delta_shares."
    )

# ---- P64 / F2.4 (2026-09-20): gross observed delta ----


def test_p64_semantica_gross_observed_delta():
    """P64 / F2.4: delta_shares es GROSS OBSERVED DELTA.

    No es delta economico. La constante lo declara explicitamente.
    """
    assert ds.DELTA_SEMANTICS_GROSS_OBSERVED is True
    assert "split" in ds.P64_EVENTS_DEFERRED
    assert "reverse_split" in ds.P64_EVENTS_DEFERRED
    assert "spin_off" in ds.P64_EVENTS_DEFERRED
    assert "merger" in ds.P64_EVENTS_DEFERRED
    assert "share_class_conversion" in ds.P64_EVENTS_DEFERRED


def test_p64_delta_no_lleva_columnas_economic_mechanical():
    """P64 / F2.4: DELTA_COLUMNS no expone economic ni mechanical.

    El sistema calcula un unico delta bruto. La separacion
    economic/mechanical queda como capacidad diferida v1.
    """
    assert "delta_shares_economic" not in ds.DELTA_COLUMNS
    assert "delta_shares_mechanical" not in ds.DELTA_COLUMNS
    assert "delta_shares" in ds.DELTA_COLUMNS


def test_p64_split_no_se_etiqueta_como_economico():
    """P64 / F2.4: escenario split 1:2 -> delta=+100 sin etiqueta economica.

    Q4 = 100 acciones
    Q1 = 200 acciones (mismo CUSIP)
    delta_shares = +100

    El sistema NO puede distinguir si ese +100 es:
      - compra de 100 acciones (economico)
      - split 1:2 (mecanico)
      - mezcla

    Por tanto:
      - delta_shares debe ser exactamente 100.
      - NO debe existir ninguna columna que lo etiquete como
        "economic" o "mechanical".
      - match_status debe ser BOTH (continuidad observada).
    """
    info_q4 = _mk_infotable([
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 1, "CUSIP": "037833100",
         "SSHPRNAMT": 100.0},
    ])
    info_q1 = _mk_infotable([
        {"ACCESSION_NUMBER": "B1", "INFOTABLE_SK": 1, "CUSIP": "037833100",
         "SSHPRNAMT": 200.0},
    ])
    sub_q4 = _mk_submission([("A1", "FM1")])
    sub_q1 = _mk_submission([("B1", "FM1")])
    idr = {"037833100": _identity_entry("037833100", canonical="equity:AAPL")}

    u_q4 = ds.compute_reported_position_units(
        info_q4, sub_q4, report_period="2025-12-31", identity_results=idr,
    )
    u_q1 = ds.compute_reported_position_units(
        info_q1, sub_q1, report_period="2026-03-31", identity_results=idr,
    )
    delta = ds.compute_delta_shares(u_q1, u_q4)

    assert len(delta) == 1
    row = delta.iloc[0]
    assert row["match_status"] == ds.STATUS_BOTH
    # El delta es bruto, +100.
    assert float(row["delta_shares"]) == 100.0
    # NO existe etiqueta economica.
    assert "delta_shares_economic" not in delta.columns
    assert "delta_shares_mechanical" not in delta.columns
    assert set(delta.columns) == set(ds.DELTA_COLUMNS)


def test_p64_cusip_change_es_identidad_no_corporate_action():
    """P64 / F2.4: CUSIP change -> continuidad de IDENTIDAD, no deteccion
    de evento societario.

    Cuando CUSIP_A (Q4) y CUSIP_B (Q1) resuelven al mismo canonical,
    el reconciliador produce BOTH. Esto demuestra continuidad de
    identidad; NO demuestra que el sistema haya identificado el evento
    societario concreto que provoco el cambio de CUSIP.

    Las dos capas son distintas:
        IDENTITY CONTINUITY  !=  CORPORATE ACTION DETECTION
    """
    info_q4 = _mk_infotable([
        {"ACCESSION_NUMBER": "A1", "INFOTABLE_SK": 1, "CUSIP": "CUSIP_OLD",
         "SSHPRNAMT": 100.0},
    ])
    info_q1 = _mk_infotable([
        {"ACCESSION_NUMBER": "B1", "INFOTABLE_SK": 1, "CUSIP": "CUSIP_NEW",
         "SSHPRNAMT": 120.0},
    ])
    sub_q4 = _mk_submission([("A1", "FM1")])
    sub_q1 = _mk_submission([("B1", "FM1")])
    idr = {
        "CUSIP_OLD": _identity_entry("CUSIP_OLD", canonical="equity:X"),
        "CUSIP_NEW": _identity_entry("CUSIP_NEW", canonical="equity:X"),
    }
    u_q4 = ds.compute_reported_position_units(
        info_q4, sub_q4, report_period="2025-12-31", identity_results=idr,
    )
    u_q1 = ds.compute_reported_position_units(
        info_q1, sub_q1, report_period="2026-03-31", identity_results=idr,
    )
    delta = ds.compute_delta_shares(u_q1, u_q4)
    assert len(delta) == 1
    row = delta.iloc[0]
    assert row["match_status"] == ds.STATUS_BOTH
    assert float(row["delta_shares"]) == 20.0
    # El sistema NO emite evento societario: no existe columna que lo
    # declare, ni constante P64_EVENTS_DETECTED.
    assert "corporate_action" not in delta.columns
    assert not hasattr(ds, "P64_EVENTS_DETECTED")
