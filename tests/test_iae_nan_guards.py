# -*- coding: utf-8 -*-
# Centinela B6-H10: iae.py con NaN/inf en campos numericos.

from src.report.iae import render_iae_section, _fmt_int


def test_fmt_int_nan_nd():
    assert _fmt_int(float("nan")) == "N/D"


def test_fmt_int_inf_nd():
    assert _fmt_int(float("inf")) == "N/D"
    assert _fmt_int(float("-inf")) == "N/D"


def test_fmt_int_none_nd():
    assert _fmt_int(None) == "N/D"


def test_fmt_int_normal_con_miles():
    assert _fmt_int(1234) == "1.234"
    assert _fmt_int(-5678) == "-5.678"


def _iae_full(**kwargs):
    d = {
        "status": "OK",
        "period_previous": "Q2-2025", "period_current": "Q3-2025",
        "nipc_total": 1000, "nipc_sole": 500, "nipc_dfnd": 300, "nipc_otr": 200,
        "catalog_coverage_declared": 0.95,
        "catalog_keys_observed": 100, "catalog_keys_total": 100,
        "coverage_status": "OK", "coverage_quality": "COMPLETE",
        "n_delta_observable": 50, "n_both": 30, "n_new": 10, "n_exit": 10,
        "n_unresolved_identity": 0,
    }
    d.update(kwargs)
    return "".join(render_iae_section(d))


def test_iae_sin_nan_literal():
    out = _iae_full(nipc_total=float("nan"), catalog_coverage_declared=float("nan"),
                    n_unresolved_identity=float("nan"))
    lineas_nan = [l for l in out.split("\n") if "nan" in l.lower()]
    assert not lineas_nan, "NaN literal: " + str(lineas_nan)


def test_iae_cobertura_nan_pinta_nd():
    out = _iae_full(catalog_coverage_declared=float("nan"))
    assert "(N/D)" in out
    assert "nan%" not in out


def test_iae_nipc_nan_pinta_nd():
    out = _iae_full(nipc_total=float("nan"))
    assert "| Total | N/D |" in out


def test_iae_valores_normales_intactos():
    out = _iae_full()
    assert "| Total | 1.000 |" in out
    assert "95.00%" in out
    assert "UNRESOLVED_IDENTITY: 0" in out
