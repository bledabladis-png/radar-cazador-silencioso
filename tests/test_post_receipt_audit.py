"""REP-AUDIT-20261018: auditoria post-receipt.

Valida que scripts/post_receipt_audit.py detecta MISSING cuando un
receipt declara written=True con date=D pero el CSV no tiene fila D.
Captura el caso del bug 09-oct (receipt emitido sin la fila real).

Hermeticos: tmp_path, sin tocar outputs/ ni data/ reales.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.post_receipt_audit as pra


def _write_csv(path: Path, dates: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"date": dates, "value": [1] * len(dates)})
    df.to_csv(path, index=False)


def _receipt(name, written, date=None, path="", rows=1):
    return {"name": name, "written": written, "reason": "OK" if written else "SKIP_CONDITION",
            "date": date, "path": path, "rows": rows}


# ---------- _has_row ----------

def test_has_row_true(tmp_path):
    p = tmp_path / "x.csv"
    _write_csv(p, ["2026-10-14", "2026-10-15", "2026-10-16"])
    assert pra._has_row(p, "2026-10-15") is True


def test_has_row_false(tmp_path):
    p = tmp_path / "x.csv"
    _write_csv(p, ["2026-10-14", "2026-10-15"])
    assert pra._has_row(p, "2026-10-16") is False


def test_has_row_no_existe(tmp_path):
    p = tmp_path / "no_existe.csv"
    assert pra._has_row(p, "2026-10-16") is False


# ---------- audit ----------

def test_audit_ok(tmp_path):
    p = tmp_path / "x.csv"
    _write_csv(p, ["2026-10-16"])
    receipts = [_receipt("x", True, "2026-10-16", str(p))]
    df = pra.audit(receipts, "2026-10-16")
    assert df.iloc[0]["status"] == "OK"


def test_audit_missing(tmp_path):
    """Caso bug 09-oct: written=True con date=D, CSV no tiene D."""
    p = tmp_path / "x.csv"
    _write_csv(p, ["2026-10-14", "2026-10-15"])
    receipts = [_receipt("x", True, "2026-10-16", str(p))]
    df = pra.audit(receipts, "2026-10-16")
    assert df.iloc[0]["status"] == "MISSING"


def test_audit_csv_missing(tmp_path):
    receipts = [_receipt("x", True, "2026-10-16", str(tmp_path / "no.csv"))]
    df = pra.audit(receipts, "2026-10-16")
    assert df.iloc[0]["status"] == "CSV_MISSING"


def test_audit_skip_no_verifica(tmp_path):
    """HOLIDAY_MODE o candado B5: written=False no se audita."""
    receipts = [_receipt("macro_regime", False, None, "")]
    df = pra.audit(receipts, "2026-10-16")
    assert df.iloc[0]["status"] == "SKIP"


def test_audit_no_date_check_snapshot(tmp_path):
    """Snapshots con date=None no se auditan."""
    p = tmp_path / "x.csv"
    _write_csv(p, ["2026-10-16"])
    receipts = [_receipt("sector_rank_deltas", True, None, str(p))]
    df = pra.audit(receipts, "2026-10-16")
    assert df.iloc[0]["status"] == "NO_DATE_CHECK"


def test_audit_sin_receipts():
    df = pra.audit([], "2026-10-16")
    assert df.empty


# ---------- _append_csv ----------

def test_append_csv_nuevo(tmp_path):
    p = tmp_path / "audit.csv"
    df = pd.DataFrame([{"target_session": "2026-10-16", "writer": "x",
                        "status": "OK", "receipt_date": "2026-10-16", "csv_path": "p"}],
                      columns=pra.AUDIT_COLUMNS)
    pra._append_csv(df, p)
    assert p.exists()
    result = pd.read_csv(p)
    assert len(result) == 1


def test_append_csv_dedup(tmp_path):
    """Rerun con misma (target_session, writer) no duplica."""
    p = tmp_path / "audit.csv"
    df1 = pd.DataFrame([{"target_session": "2026-10-16", "writer": "x",
                         "status": "OK", "receipt_date": "2026-10-16", "csv_path": "p"}],
                       columns=pra.AUDIT_COLUMNS)
    pra._append_csv(df1, p)
    pra._append_csv(df1, p)
    assert len(pd.read_csv(p)) == 1


# ---------- main ----------

def test_main_sin_json(tmp_path, monkeypatch):
    monkeypatch.setattr(pra, "RECEIPTS_PATH", tmp_path / "no.json")
    monkeypatch.setattr(pra, "AUDIT_CSV", tmp_path / "audit.csv")
    assert pra.main() == 0


def test_main_json_vacio(tmp_path, monkeypatch):
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps({"receipts": []}), encoding="utf-8")
    monkeypatch.setattr(pra, "RECEIPTS_PATH", rp)
    monkeypatch.setattr(pra, "AUDIT_CSV", tmp_path / "audit.csv")
    assert pra.main() == 0