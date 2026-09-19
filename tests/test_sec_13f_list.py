# -*- coding: utf-8 -*-
"""Tests de sec_13f.identity.sec13f_list. Sin red."""
import pandas as pd
import pytest

from src.institutional_accumulation.sec_13f.identity import sec13f_list as sl


# ---- constructor de lineas sinteticas ----

def _mk_line(cusip="037833100", option=" ", name="APPLE INC",
             desc="COM", status="   ", pos80="E"):
    """Construye una linea fixed-width 80 con posiciones canonicas."""
    # 1-9 CUSIP (9)
    # 10 option (1)
    # 11-40 issuer_name (30)
    # 41-67 issuer_description (27)
    # 68-70 status (3)
    # 71-79 blank (9)
    # 80 misc (1)
    cusip9 = cusip.ljust(9)[:9]
    name30 = name.ljust(30)[:30]
    desc27 = desc.ljust(27)[:27]
    status3 = status.ljust(3)[:3]
    blank9 = " " * 9
    return cusip9 + option + name30 + desc27 + status3 + blank9 + pos80


# ---- constantes ----

def test_estados_definidos():
    assert sl.STATUS_ADDED == "ADDED"
    assert sl.STATUS_DELETED == "DELETED"
    assert sl.STATUS_ACTIVE == "ACTIVE"
    assert sl.STATUS_NOT_IN_LIST == "NOT_IN_LIST"
    assert sl.STATUS_CONFLICT == "CONFLICT"
    assert set(sl.ALL_STATUSES) == {
        "ADDED", "DELETED", "ACTIVE", "NOT_IN_LIST", "CONFLICT"
    }
    assert sl.ELIGIBLE_STATES == ("ADDED", "ACTIVE")


def test_layout_constantes():
    assert sl.LINE_WIDTH == 80
    assert sl.POS_CUSIP == (0, 9)
    assert sl.POS_OPTION == 9
    assert sl.POS_ISSUER_NAME == (10, 40)
    assert sl.POS_ISSUER_DESC == (40, 67)
    assert sl.POS_STATUS == (67, 70)
    assert sl.POS_MISC == 79


# ---- parse_line ----

def test_parse_line_basico():
    line = _mk_line()
    p = sl.parse_line(line)
    assert p["cusip"] == "037833100"
    assert p["issuer_name"] == "APPLE INC"
    assert p["issuer_description"] == "COM"
    assert p["option_indicator"] == " "
    assert p["status_raw"] == "   "
    assert p["status"] == sl.STATUS_ACTIVE
    assert len(p["raw_line"]) == 80


def test_parse_line_status_added():
    p = sl.parse_line(_mk_line(status="*A*"))
    assert p["status_raw"] == "*A*"
    assert p["status"] == sl.STATUS_ADDED


def test_parse_line_status_deleted():
    p = sl.parse_line(_mk_line(status="*D*"))
    assert p["status"] == sl.STATUS_DELETED


def test_parse_line_status_desconocido_devuelve_none():
    p = sl.parse_line(_mk_line(status="XYZ"))
    assert p["status"] is None


def test_parse_line_vacio_devuelve_none():
    assert sl.parse_line("") is None
    assert sl.parse_line("   ") is None
    assert sl.parse_line(None) is None


def test_parse_line_ignora_pos_80():
    a = sl.parse_line(_mk_line(pos80="E"))
    b = sl.parse_line(_mk_line(pos80="X"))
    assert a["cusip"] == b["cusip"]
    assert a["status"] == b["status"]
    assert a["issuer_name"] == b["issuer_name"]
    # raw_line difiere en el ultimo char, pero ningun campo semantico lo usa


def test_parse_line_padding_si_short():
    """P51: linea corta (< 70): no crashea, se preserva como anomalia."""
    short = "037833100" + " " + "APPLE INC".ljust(30)  # 40 chars
    p = sl.parse_line(short)
    assert p is not None
    assert p["cusip"] == "037833100"
    assert p["status"] is None


def test_parse_line_demasiado_corta_status_none():
    """P51: linea < 70 chars -> status=None (anomalia), no ACTIVE."""
    line_50 = "A" * 50
    p = sl.parse_line(line_50)
    assert p is not None
    assert p["status"] is None
    assert p["status_raw"] is None
    assert p["raw_line"] == line_50


def test_parse_line_exactamente_70_chars_parsable():
    """P51: linea de exactamente 70 chars se parsea con normalidad."""
    line = ("037833100" + " " + "APPLE INC".ljust(30)
            + "COM".ljust(27) + "   ")
    assert len(line) == 70
    p = sl.parse_line(line)
    assert p is not None
    assert p["cusip"] == "037833100"
    assert p["status"] == sl.STATUS_ACTIVE


# ---- parse_official_list_text ----

def test_parse_official_list_text_varias_lineas():
    text = "\n".join([
        _mk_line(cusip="037833100", name="APPLE INC", status="   "),
        _mk_line(cusip="594918104", name="MICROSOFT CORP", status="*A*"),
        _mk_line(cusip="023135106", name="AMAZON COM INC", status="*D*"),
        "",
    ])
    df = sl.parse_official_list_text(text)
    assert len(df) == 3
    assert df.iloc[0]["cusip"] == "037833100"
    assert df.iloc[1]["status"] == sl.STATUS_ADDED
    assert df.iloc[2]["status"] == sl.STATUS_DELETED


def test_parse_official_list_text_preserva_order_y_line_no():
    text = "\n".join([
        "",  # linea 1 vacia
        _mk_line(cusip="037833100"),  # linea 2
        "   ",  # linea 3 vacia
        _mk_line(cusip="594918104"),  # linea 4
    ])
    df = sl.parse_official_list_text(text)
    assert df.iloc[0]["_line_no"] == 2
    assert df.iloc[1]["_line_no"] == 4
    assert df.iloc[0]["_order"] == 0
    assert df.iloc[1]["_order"] == 1


# ---- load_official_list ----

def test_load_official_list_no_existe(tmp_path):
    with pytest.raises(FileNotFoundError, match="no existe"):
        sl.load_official_list(tmp_path / "no_existe.txt")


def test_load_official_list_roundtrip(tmp_path):
    p = tmp_path / "list.txt"
    p.write_text(_mk_line(cusip="037833100") + "\n", encoding="utf-8")
    df = sl.load_official_list(p)
    assert len(df) == 1
    assert df.iloc[0]["cusip"] == "037833100"


# ---- resolve_eligibility ----

def test_resolve_not_in_list():
    df = sl.parse_official_list_text(_mk_line(cusip="037833100"))
    r = sl.resolve_eligibility(["999999999"], df)
    assert r["999999999"]["status"] == sl.STATUS_NOT_IN_LIST
    assert r["999999999"]["eligible"] is False


def test_resolve_active():
    df = sl.parse_official_list_text(_mk_line(cusip="037833100", status="   "))
    r = sl.resolve_eligibility(["037833100"], df)
    assert r["037833100"]["status"] == sl.STATUS_ACTIVE
    assert r["037833100"]["eligible"] is True


def test_resolve_added():
    df = sl.parse_official_list_text(_mk_line(cusip="037833100", status="*A*"))
    r = sl.resolve_eligibility(["037833100"], df)
    assert r["037833100"]["status"] == sl.STATUS_ADDED
    assert r["037833100"]["eligible"] is True


def test_resolve_deleted():
    df = sl.parse_official_list_text(_mk_line(cusip="037833100", status="*D*"))
    r = sl.resolve_eligibility(["037833100"], df)
    assert r["037833100"]["status"] == sl.STATUS_DELETED
    assert r["037833100"]["eligible"] is False


def test_resolve_conflict_a_y_d():
    """Mismo CUSIP con *A* y *D* -> CONFLICT (no 'ultimo gana')."""
    text = "\n".join([
        _mk_line(cusip="037833100", status="*D*"),
        _mk_line(cusip="037833100", status="*A*"),
    ])
    df = sl.parse_official_list_text(text)
    r = sl.resolve_eligibility(["037833100"], df)
    assert r["037833100"]["status"] == sl.STATUS_CONFLICT
    assert r["037833100"]["eligible"] is False


def test_resolve_conflict_por_status_desconocido():
    """Linea con status raw invalido -> status=None -> CONFLICT."""
    text = _mk_line(cusip="037833100", status="XYZ")
    df = sl.parse_official_list_text(text)
    r = sl.resolve_eligibility(["037833100"], df)
    assert r["037833100"]["status"] == sl.STATUS_CONFLICT


def test_resolve_official_df_empty():
    df = pd.DataFrame()
    r = sl.resolve_eligibility(["037833100"], df)
    assert r["037833100"]["status"] == sl.STATUS_NOT_IN_LIST
    assert r["037833100"]["eligible"] is False


def test_resolve_official_df_none():
    r = sl.resolve_eligibility(["037833100"], None)
    assert r["037833100"]["status"] == sl.STATUS_NOT_IN_LIST


def test_resolve_option_indicator_capturado():
    text = _mk_line(cusip="037833100", option="*")
    df = sl.parse_official_list_text(text)
    r = sl.resolve_eligibility(["037833100"], df)
    assert r["037833100"]["option_indicator"] == "*"


# ---- compute_eligibility_coverage ----

def test_compute_eligibility_coverage_conteos():
    text = "\n".join([
        _mk_line(cusip="A00000000", status="   "),  # ACTIVE
        _mk_line(cusip="B00000000", status="*A*"),  # ADDED
        _mk_line(cusip="C00000000", status="*D*"),  # DELETED
    ])
    df = sl.parse_official_list_text(text)
    cusips = ["A00000000", "B00000000", "C00000000", "D99999999"]
    m = sl.compute_eligibility_coverage(cusips, df)
    assert m["n_total"] == 4
    assert m["n_active"] == 1
    assert m["n_added"] == 1
    assert m["n_deleted"] == 1
    assert m["n_not_in_list"] == 1
    assert m["n_eligible"] == 2
    assert 0.0 < m["pct_eligible"] < 1.0
    assert abs(m["pct_eligible"] - 0.5) < 1e-9


def test_compute_eligibility_coverage_vacio():
    m = sl.compute_eligibility_coverage([], pd.DataFrame())
    assert m["n_total"] == 0
    assert m["pct_eligible"] == 0.0