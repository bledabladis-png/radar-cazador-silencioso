"""Tests de scripts/regenerate_radar_catalog.py (sin red)."""
from __future__ import annotations

import json
import pandas as pd

from scripts import regenerate_radar_catalog as rrc
from src.institutional_accumulation.identity.radar_target_catalog import (
    SOURCE_NAME,
)


def _catalog(n: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "radar_ticker": f"T{i:03d}",
            "figi": f"BBG{i}",
            "share_class_figi": f"SC{i}",
            "composite_figi": f"BBG{i}",
            "ticker_from_openfigi": f"T{i:03d}",
            "name": f"NAME{i}",
            "security_type": "Common Stock",
            "market_sector": "Equity",
            "exch_code": "US",
            "source": SOURCE_NAME,
            "source_date": "2026-01-01",
            "status": "OK",
        })
    return pd.DataFrame(rows)


def _hit_ok(tk: str) -> dict:
    return {"ok": True, "data": [{
        "figi": f"BBG_{tk}",
        "shareClassFIGI": f"SC_{tk}",
        "compositeFIGI": f"BBG_{tk}",
        "ticker": tk,
        "name": f"NAME_{tk}",
        "securityType": "Common Stock",
        "marketSector": "Equity",
        "exchCode": "US",
    }], "error": None}


def _hit_miss() -> dict:
    return {"ok": False, "data": None, "error": "No identifier found."}


# --- _compute_diff ---

def test_diff_sin_cambios():
    cat = _catalog(3)
    d = rrc._compute_diff(["T000", "T001", "T002"], cat)
    assert d["nuevos"] == []
    assert d["desaparecidos"] == []
    assert len(d["estables"]) == 3


def test_diff_nuevos_y_desaparecidos():
    cat = _catalog(3)  # T000 T001 T002
    d = rrc._compute_diff(["T000", "T003"], cat)
    assert d["nuevos"] == ["T003"]
    assert d["desaparecidos"] == ["T001", "T002"]
    assert d["estables"] == ["T000"]


def test_diff_catalogo_vacio():
    d = rrc._compute_diff(["A", "B"], pd.DataFrame(columns=list(rrc.COLUMNS)))
    assert d["nuevos"] == ["A", "B"]
    assert d["desaparecidos"] == []


# --- _next_snapshot_id ---

def test_next_snapshot_id_desde_vacio(tmp_path):
    assert rrc._next_snapshot_id(tmp_path, "2026-09-24") == "2026-09-24_01"


def test_next_snapshot_id_incremental(tmp_path):
    (tmp_path / "snapshot_2026-09-24_01.csv").write_text("")
    (tmp_path / "snapshot_2026-09-24_02.csv").write_text("")
    assert rrc._next_snapshot_id(tmp_path, "2026-09-24") == "2026-09-24_03"


# --- _build_updated_catalog ---

def test_build_updated_catalog_sin_nuevos():
    cat = _catalog(3)
    out = rrc._build_updated_catalog(cat, [], {}, source_date="2026-09-24")
    assert len(out) == 3
    assert (out["radar_ticker"] == cat["radar_ticker"]).all()


def test_build_updated_catalog_con_nuevos():
    cat = _catalog(2)  # T000 T001
    hits = {"T999": _hit_ok("T999")}
    out = rrc._build_updated_catalog(cat, ["T999"], hits, source_date="2026-09-24")
    assert len(out) == 3
    assert "T999" in set(out["radar_ticker"])


# --- _close_active_snapshot ---

def test_close_active_snapshot():
    m = {"snapshots": [
        {"version_id": "v1", "valid_to": "2026-01-01"},
        {"version_id": "v2", "valid_to": None},
    ]}
    rrc._close_active_snapshot(m, "2026-09-24")
    assert m["snapshots"][0]["valid_to"] == "2026-01-01"
    assert m["snapshots"][1]["valid_to"] == "2026-09-24"


# --- main() con mocks ---

def _setup_env(tmp_path, monkeypatch):
    """Redirige todas las rutas del script a tmp_path."""
    mappings = tmp_path / "mappings"
    snapshots = mappings / "catalog_snapshots"
    mappings.mkdir()
    snapshots.mkdir()
    cat_path = mappings / "radar_target_catalog.csv"
    manifest = mappings / "catalog_manifest.json"
    _catalog(2).to_csv(cat_path, index=False)
    manifest.write_text(json.dumps({"schema_version": 1, "snapshots": []}))
    monkeypatch.setattr(rrc, "MAPPINGS", mappings)
    monkeypatch.setattr(rrc, "SNAPSHOTS", snapshots)
    monkeypatch.setattr(rrc, "MANIFEST", manifest)
    monkeypatch.setattr(rrc, "CATALOG_CSV", cat_path)
    return mappings, snapshots, manifest, cat_path


def test_main_sin_diff_no_escribe(tmp_path, monkeypatch):
    mappings, snapshots, manifest, cat_path = _setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(rrc, "load_radar_tickers", lambda p: ["T000", "T001"])
    monkeypatch.setattr("sys.argv", ["regenerate_radar_catalog.py", "--source-date", "2026-09-24"])
    before = manifest.read_text()
    assert rrc.main() == 0
    assert manifest.read_text() == before
    assert not list(snapshots.glob("*.csv"))


def test_main_aborta_si_demasiados_nuevos(tmp_path, monkeypatch):
    mappings, snapshots, manifest, cat_path = _setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(rrc, "load_radar_tickers", lambda p: ["T000", "T001"] + [f"N{i:03d}" for i in range(25)])
    monkeypatch.setattr("sys.argv", ["regenerate_radar_catalog.py", "--source-date", "2026-09-24"])
    assert rrc.main() == 1
    assert not list(snapshots.glob("*.csv"))


def test_main_aborta_si_openfigi_sin_hits(tmp_path, monkeypatch):
    mappings, snapshots, manifest, cat_path = _setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(rrc, "load_radar_tickers", lambda p: ["T000", "T001", "T999"])
    monkeypatch.setattr(rrc, "map_identifiers", lambda *a, **k: {"T999": _hit_miss()})
    monkeypatch.setattr("sys.argv", ["regenerate_radar_catalog.py", "--source-date", "2026-09-24"])
    assert rrc.main() == 1
    assert not list(snapshots.glob("*.csv"))


def test_main_con_nuevo_escribe_todo(tmp_path, monkeypatch):
    mappings, snapshots, manifest, cat_path = _setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(rrc, "load_radar_tickers", lambda p: ["T000", "T001", "T999"])
    monkeypatch.setattr(rrc, "map_identifiers", lambda *a, **k: {"T999": _hit_ok("T999")})
    monkeypatch.setattr("sys.argv", ["regenerate_radar_catalog.py", "--source-date", "2026-09-24"])
    assert rrc.main() == 0
    # snapshot escrito
    snaps = list(snapshots.glob("snapshot_2026-09-24_01.csv"))
    assert len(snaps) == 1
    assert snaps[0].with_suffix(".sha256").exists() or (snapshots / "snapshot_2026-09-24_01.sha256").exists()
    # manifest actualizado
    m = json.loads(manifest.read_text())
    assert len(m["snapshots"]) == 1
    assert m["snapshots"][0]["version_id"] == "2026-09-24_01"
    assert m["snapshots"][0]["valid_to"] is None
    # vista actualizada
    vista = pd.read_csv(cat_path, dtype=str)
    assert "T999" in set(vista["radar_ticker"])


def test_main_dry_run_no_escribe(tmp_path, monkeypatch):
    mappings, snapshots, manifest, cat_path = _setup_env(tmp_path, monkeypatch)
    monkeypatch.setattr(rrc, "load_radar_tickers", lambda p: ["T000", "T001", "T999"])
    monkeypatch.setattr(rrc, "map_identifiers", lambda *a, **k: {"T999": _hit_ok("T999")})
    monkeypatch.setattr("sys.argv", ["regenerate_radar_catalog.py", "--source-date", "2026-09-24", "--dry-run"])
    assert rrc.main() == 0
    assert not list(snapshots.glob("*.csv"))
    m = json.loads(manifest.read_text())
    assert m["snapshots"] == []


# --- _check_equity_only (chequeo defensivo post-construccion) ------------

def test_check_equity_only_ok_sin_no_equity():
    """Todos Equity -> sin avisos."""
    df = pd.DataFrame([
        {"radar_ticker": "AAPL", "status": "OK",
         "market_sector": "Equity", "security_type": "Common Stock"},
        {"radar_ticker": "AMT", "status": "OK",
         "market_sector": "Equity", "security_type": "REIT"},
    ])
    assert rrc._check_equity_only(df) == []


def test_check_equity_only_detecta_etf():
    """Ticker con market_sector=ETF -> aparece en el listado."""
    df = pd.DataFrame([
        {"radar_ticker": "AAPL", "status": "OK",
         "market_sector": "Equity", "security_type": "Common Stock"},
        {"radar_ticker": "XLK", "status": "OK",
         "market_sector": "Equity", "security_type": "ETF"},  # market_sector Equity
        {"radar_ticker": "SPY", "status": "OK",
         "market_sector": "ETF", "security_type": "ETF"},     # sector != Equity
    ])
    bad = rrc._check_equity_only(df)
    assert len(bad) == 1
    assert bad[0]["radar_ticker"] == "SPY"


def test_check_equity_only_ignora_status_no_ok():
    """Un no-equity con status != OK no genera aviso (no es canonical)."""
    df = pd.DataFrame([
        {"radar_ticker": "SPY", "status": "MISS",
         "market_sector": "ETF", "security_type": "ETF"},
    ])
    assert rrc._check_equity_only(df) == []


def test_check_equity_only_vacio():
    assert rrc._check_equity_only(pd.DataFrame()) == []
    assert rrc._check_equity_only(pd.DataFrame(columns=list(rrc.COLUMNS))) == []
