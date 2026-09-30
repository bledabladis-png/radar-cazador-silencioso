"""Tests unitarios de pipeline_contractual (E.2.d).

Cubren las piezas puras: ticker_of y _subset_universe.
La orquestacion (run_contractual_nipc) se cubre por E2E via
scripts/iae_contractual_nipc_e2e.py (reconcilia contra baseline
current.json; ver correccion 2026-09-29 en golden/README.md).
"""
from __future__ import annotations

import pandas as pd

from src.institutional_accumulation.pipeline_contractual import (
    ticker_of,
    _subset_universe,
)
from src.institutional_accumulation.identity.target_builder import (
    TargetUniverse,
)


def _make_universe(keys):
    """TargetUniverse minimo con coherencia ticker/figi/row_uid."""
    keys = list(keys)
    return TargetUniverse(
        version_id="v_test",
        period_end="2026-03-31",
        catalog_version_id="cat_test",
        catalog_sha256="a" * 64,
        declared_keys=set(keys),
        ticker_by_key={k: "TK_" + k for k in keys},
        figi_by_key={k: "FG_" + k for k in keys},
        row_uid_by_key={k: "UID_" + k for k in keys},
        key_by_row_uid={"UID_" + k: k for k in keys},
    )

def test_ticker_of_equity_extrae_ticker():
    s = pd.Series(["equity:AAPL", "equity:MSFT", "equity:GOOGL"])
    out = ticker_of(s)
    assert list(out) == ["AAPL", "MSFT", "GOOGL"]


def test_ticker_of_figi_devuelve_none():
    s = pd.Series(["figi:BBG001S5N8V8", "figi:BBG009S3NB21"])
    out = ticker_of(s)
    assert out.isna().all()


def test_ticker_of_nan_y_vacios():
    s = pd.Series([None, "", "   ", float("nan")])
    out = ticker_of(s)
    assert out.isna().all()


def test_ticker_of_mixto():
    s = pd.Series(["equity:AAPL", "figi:BBG001", None, "cusip:037833100"])
    out = ticker_of(s)
    assert out.iloc[0] == "AAPL"
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[2])
    assert pd.isna(out.iloc[3])


def test_subset_universe_reduce_keys():
    u = _make_universe(["k1", "k2", "k3"])
    u2 = _subset_universe(u, {"k1", "k2"})
    assert u2.declared_keys == {"k1", "k2"}
    assert set(u2.ticker_by_key.keys()) == {"k1", "k2"}
    assert set(u2.figi_by_key.keys()) == {"k1", "k2"}
    assert set(u2.row_uid_by_key.keys()) == {"k1", "k2"}
    assert u2.key_by_row_uid == {"UID_k1": "k1", "UID_k2": "k2"}


def test_subset_universe_preserva_metadata():
    u = _make_universe(["k1", "k2", "k3"])
    u2 = _subset_universe(u, {"k2"})
    assert u2.version_id == u.version_id
    assert u2.period_end == u.period_end
    assert u2.catalog_version_id == u.catalog_version_id
    assert u2.catalog_sha256 == u.catalog_sha256
    assert u2.ticker_by_key == {"k2": "TK_k2"}


# ---------- D18 (2026-09-30): build_identities ----------
# Cubre las 2 ramas del fichero de equivalencia (existe vs no existe).
# La orquestacion run_contractual_nipc sigue cubierta por E2E; aqui
# solo las piezas con ramificacion clara.

def test_build_identities_sin_equivalence_usa_none(tmp_path, monkeypatch):
    """Sin cusip_equivalence.csv -> eq=None, se pasa None a resolver."""
    from unittest.mock import patch as _patch
    from src.institutional_accumulation import pipeline_contractual as pc

    snap = {"INFOTABLE": pd.DataFrame({"CUSIP": ["037833100", "594918104"]})}
    mp = tmp_path
    # NO crear cusip_equivalence.csv -> rama else.

    captured = {}

    def fake_resolve(cusips, iso, *, equivalence_df, crosswalk_internal_df,
                     figi_lookup):
        captured["cusips"] = sorted(cusips)
        captured["equivalence_df"] = equivalence_df
        captured["iso"] = iso
        return {"resolved": True}

    with _patch.object(pc, "resolve_batch_identities", fake_resolve), \
         _patch.object(pc, "load_crosswalk_internal", lambda: "CW"):
        result = pc.build_identities(snap, "2026-03-31", mappings_dir=mp)

    assert result == {"resolved": True}
    assert captured["cusips"] == ["037833100", "594918104"]
    assert captured["equivalence_df"] is None
    assert captured["iso"] == "2026-03-31"


def test_build_identities_con_equivalence_la_carga(tmp_path):
    """Con cusip_equivalence.csv -> load_cusip_equivalence(eqp)."""
    from unittest.mock import patch as _patch
    from src.institutional_accumulation import pipeline_contractual as pc

    mp = tmp_path
    eqp = mp / "cusip_equivalence.csv"
    eqp.write_text("cusip,valid_from,valid_to\n037833100,2025-01-01,2026-12-31\n",
                   encoding="utf-8")

    snap = {"INFOTABLE": pd.DataFrame({"CUSIP": ["037833100"]})}

    loaded_paths = []
    def fake_load(path):
        loaded_paths.append(str(path))
        return "EQ_DF"

    captured = {}
    def fake_resolve(cusips, iso, *, equivalence_df, crosswalk_internal_df,
                     figi_lookup):
        captured["equivalence_df"] = equivalence_df
        return {"ok": True}

    with _patch.object(pc, "load_cusip_equivalence", fake_load), \
         _patch.object(pc, "resolve_batch_identities", fake_resolve), \
         _patch.object(pc, "load_crosswalk_internal", lambda: "CW"):
        pc.build_identities(snap, "2026-03-31", mappings_dir=mp)

    assert loaded_paths == [str(eqp)]
    assert captured["equivalence_df"] == "EQ_DF"


def test_build_identities_dedup_cusips(tmp_path):
    """CUSIPs duplicados en INFOTABLE se pasan como unicos."""
    from unittest.mock import patch as _patch
    from src.institutional_accumulation import pipeline_contractual as pc

    snap = {"INFOTABLE": pd.DataFrame({"CUSIP": ["037833100", "037833100", "594918104"]})}
    mp = tmp_path

    captured = {}
    def fake_resolve(cusips, iso, *, equivalence_df, crosswalk_internal_df,
                     figi_lookup):
        captured["cusips"] = list(cusips)
        return {}

    with _patch.object(pc, "resolve_batch_identities", fake_resolve), \
         _patch.object(pc, "load_crosswalk_internal", lambda: "CW"):
        pc.build_identities(snap, "2026-03-31", mappings_dir=mp)

    assert sorted(captured["cusips"]) == ["037833100", "594918104"]


# ---------- D18: load_canonical ----------
# Lee los 7 TSVs, aplica filter_by_period y apply_amendments.
# Test con parquets sinteticos + monkeypatch de las 2 funciones.
# NO cubre run_contractual_nipc (orquestador E2E por diseno).

_TSVS = ("SUBMISSION", "COVERPAGE", "SUMMARYPAGE", "OTHERMANAGER",
         "OTHERMANAGER2", "SIGNATURE", "INFOTABLE")


def _write_canonical_parquets(base, folder):
    """Escribe los 7 parquets minimos en base/folder/."""
    d = base / folder
    d.mkdir(parents=True, exist_ok=True)
    for n in _TSVS:
        pd.DataFrame({"dummy": [1, 2]}).to_parquet(d / (n + ".parquet"))


def test_load_canonical_lee_7_tsvs_y_aplica_pipeline(tmp_path):
    """load_canonical: lee los 7 parquets + filter + apply_amendments."""
    from unittest.mock import patch as _patch
    from src.institutional_accumulation import pipeline_contractual as pc

    _write_canonical_parquets(tmp_path, "2026Q1")

    called = {}

    def fake_filter(dfs, iso):
        called["filter_iso"] = iso
        called["filter_keys"] = sorted(dfs.keys())
        return {**dfs, "_filtered": True}

    def fake_amendments(f, *, period):
        called["amend_period"] = period
        return {"canonical_snapshot": "SNAP_" + period}

    with _patch.object(pc, "filter_by_period", fake_filter), \
         _patch.object(pc, "apply_amendments", fake_amendments):
        result = pc.load_canonical("2026Q1", "2026-03-31", data_dir=tmp_path)

    assert result == "SNAP_2026-03-31"
    assert called["filter_iso"] == "2026-03-31"
    assert called["amend_period"] == "2026-03-31"
    assert called["filter_keys"] == sorted(_TSVS)
