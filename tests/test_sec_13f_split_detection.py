"""Tests de split_detection (2026-11-05)."""
import pandas as pd

from src.institutional_accumulation.aggregation.split_detection import (
    detect_splits, adjust_units_for_splits, _is_split,
)


def _mk_infotable(rows):
    """rows: list of (CUSIP, SSHPRNAMT, VALUE, TITLEOFCLASS='COM')."""
    data = []
    for r in rows:
        cusip, shares, value = r[0], r[1], r[2]
        title = r[3] if len(r) > 3 else "COM"
        data.append({
            "CUSIP": cusip, "SSHPRNAMT": float(shares), "VALUE": float(value),
            "SSHPRNAMTTYPE": "SH", "PUTCALL": None, "TITLEOFCLASS": title,
        })
    return pd.DataFrame(data)


# ---------- _is_split unitario ----------

def test_is_split_directo_2x():
    assert _is_split(2.0, 2.0, 100, 100) == 2.0

def test_is_split_directo_5x():
    assert _is_split(5.0, 5.1, 100, 100) == 5.0

def test_is_split_no_por_ratio_disolado():
    # Precio sube 2x pero shares estables: no es split.
    assert _is_split(2.0, 1.0, 100, 100) is None

def test_is_split_no_por_shares_disolado():
    # Shares suben 2x pero precio estable: crecimiento organico, no split.
    assert _is_split(1.0, 2.0, 100, 100) is None

def test_is_split_reverse_exige_min_filers():
    # Reverse 0.5 con pocos filers: rechazado.
    assert _is_split(0.5, 0.5, 5, 5) is None

def test_is_split_reverse_con_filers_suficientes():
    assert _is_split(0.5, 0.5, 100, 100) == 0.5


# ---------- detect_splits ----------

def _buildup_split_directo(k=24.0, n=100):
    """Prev: implied 24*precio, curr: implied precio. Shares se multiplican por K."""
    rows_prev, rows_curr = [], []
    for i in range(n):
        rows_prev.append(("09857L108", 10.0, 10.0 * 100.0 * k))  # implied = 100*k
        rows_curr.append(("09857L108", 10.0 * k, 10.0 * k * 100.0))  # implied = 100
    return _mk_infotable(rows_prev), _mk_infotable(rows_curr)


def test_detecta_split_real():
    prev, curr = _buildup_split_directo(k=24.0, n=100)
    splits = detect_splits(prev, curr)
    assert splits.get("09857L108") == 24.0


def test_no_detecta_crecimiento_organico():
    # Shares 2x, precio 1x: no split.
    prev = _mk_infotable([("AAA", 100, 100 * 10.0) for _ in range(50)])
    curr = _mk_infotable([("AAA", 200, 200 * 10.0) for _ in range(50)])
    assert "AAA" not in detect_splits(prev, curr)


def test_no_detecta_etf_apalancado():
    # TITLEOFCLASS contiene "2X LONG": excluido aunque ratios coincidan.
    rows_prev = [("XXX", 10.0, 10.0 * 100.0 * 5, "2X LONG") for _ in range(50)]
    rows_curr = [("XXX", 10.0 * 5, 10.0 * 5 * 100.0, "2X LONG") for _ in range(50)]
    assert "XXX" not in detect_splits(_mk_infotable(rows_prev), _mk_infotable(rows_curr))


def test_no_detecta_bono():
    # TITLEOFCLASS contiene "NOTE": excluido.
    rows_prev = [("YYY", 10.0, 10.0 * 100.0 * 2, "NOTE 2.250%") for _ in range(50)]
    rows_curr = [("YYY", 10.0 * 2, 10.0 * 2 * 100.0, "NOTE 2.250%") for _ in range(50)]
    assert "YYY" not in detect_splits(_mk_infotable(rows_prev), _mk_infotable(rows_curr))


def test_reverse_split_con_filers_suficientes():
    n = 100
    rows_prev = [("ZZZ", 100.0, 100.0 * 100.0) for _ in range(n)]
    rows_curr = [("ZZZ", 100.0 * 0.5, 100.0 * 0.5 * 200.0) for _ in range(n)]
    splits = detect_splits(_mk_infotable(rows_prev), _mk_infotable(rows_curr))
    assert splits.get("ZZZ") == 0.5


def test_reverse_split_con_pocos_filers_no_detecta():
    n = 10
    rows_prev = [("WWW", 100.0, 100.0 * 100.0) for _ in range(n)]
    rows_curr = [("WWW", 50.0, 50.0 * 200.0) for _ in range(n)]
    assert "WWW" not in detect_splits(_mk_infotable(rows_prev), _mk_infotable(rows_curr))


# ---------- adjust_units_for_splits ----------

def test_adjust_units_multiplica_por_k():
    units = pd.DataFrame({
        "observed_security_key": ["cusip:09857L108", "cusip:OTHER"],
        "sshprnamt_total": [1000.0, 500.0],
    })
    out = adjust_units_for_splits(units, {"09857L108": 24.0})
    assert out.loc[0, "sshprnamt_total"] == 24000.0
    assert out.loc[1, "sshprnamt_total"] == 500.0  # sin cambio


def test_adjust_units_no_modifica_input():
    units = pd.DataFrame({
        "observed_security_key": ["cusip:X"],
        "sshprnamt_total": [100.0],
    })
    adjust_units_for_splits(units, {"X": 2.0})
    assert units.loc[0, "sshprnamt_total"] == 100.0
