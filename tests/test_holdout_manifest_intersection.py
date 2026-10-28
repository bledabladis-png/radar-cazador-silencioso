"""Tests de los scripts audit/holdout_manifest.py y holdout_intersection.py.

Cubren H-01 (huella reproducible) y H-03 (interseccion de universos)
del dictamen SOW-HOLDOUT-IMMUTABLE 2026-10-28.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.audit import holdout_manifest as hm
from scripts.audit import holdout_intersection as hi

# Los parquets reservados estan gitignored. En CI no existen.
# Tests que hashean/leen los parquets directamente se skipan alli.
_PARQUETS = (
    ROOT / "data" / "stock_prices_reserved_cohort.parquet",
    ROOT / "data" / "stock_prices_reserved_r3000.parquet",
)

def _parquets_disponibles():
    return all(p.exists() for p in _PARQUETS)


_skip_sin_parquets = pytest.mark.skipif(
    not _parquets_disponibles(),
    reason="parquets reservados gitignored (no disponibles en CI)",
)


# --- H-01: manifest ---

def test_manifest_existe_y_estructura():
    """El manifest real del repo tiene los 2 holdouts con sus hashes."""
    p = ROOT / "data" / "mappings" / "holdout_manifest.json"
    assert p.exists(), "holdout_manifest.json no generado"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d.get("schema_version") == 1
    holdouts = d.get("holdouts") or {}
    assert set(holdouts.keys()) == {"cohort", "r3000"}
    for key, entry in holdouts.items():
        assert entry["sha256"], key + " sin sha256"
        assert len(entry["sha256"]) == 64, key + " sha256 no es 64 hex"
        assert entry["content_sha256"], key + " sin content_sha256"
        assert entry["tickers_sha256"], key + " sin tickers_sha256"
        assert entry["n_tickers"] > 0, key + " sin tickers"
        assert entry["date_min"] < entry["date_max"], key + " rango invalido"
        assert set(entry["fields"]) >= {"Open", "High", "Low", "Close"}


def test_manifest_cohort_valores():
    """Valores conocidos de cohort (268 tickers, 2015-2026)."""
    p = ROOT / "data" / "mappings" / "holdout_manifest.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    c = d["holdouts"]["cohort"]
    assert c["n_tickers"] == 268
    assert c["date_min"] == "2015-01-02"
    assert c["date_max"] == "2026-10-23"


def test_manifest_r3000_valores():
    """Valores conocidos de r3000 (3423 tickers, 2015-2026)."""
    p = ROOT / "data" / "mappings" / "holdout_manifest.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    r = d["holdouts"]["r3000"]
    assert r["n_tickers"] == 3423
    assert r["date_min"] == "2015-01-02"
    assert r["date_max"] == "2026-10-23"


def test_canonical_sha256_estable():
    """El hash canonico no depende del orden de fields/tickers."""
    meta_a = {
        "date_min": "2015-01-02", "date_max": "2026-10-23",
        "n_rows": 100, "n_tickers": 2, "n_fields": 2,
        "fields": ["Close", "Open"], "tickers": ["AAA", "BBB"],
    }
    meta_b = {
        "date_min": "2015-01-02", "date_max": "2026-10-23",
        "n_rows": 100, "n_tickers": 2, "n_fields": 2,
        "fields": ["Open", "Close"], "tickers": ["BBB", "AAA"],
    }
    assert hm._canonical_sha256(meta_a) == hm._canonical_sha256(meta_b)


def test_canonical_sha256_cambia_con_un_ticker():
    """Anadir un ticker cambia el hash canonico (no solo el sha256 del fichero)."""
    base = {
        "date_min": "2015-01-02", "date_max": "2026-10-23",
        "n_rows": 100, "n_tickers": 2, "n_fields": 2,
        "fields": ["Close"], "tickers": ["AAA", "BBB"],
    }
    otro = dict(base)
    otro["tickers"] = ["AAA", "BBB", "CCC"]
    otro["n_tickers"] = 3
    assert hm._canonical_sha256(base) != hm._canonical_sha256(otro)


@_skip_sin_parquets
def test_verify_actual_pasa():
    """El manifest actual coincide con los parquets. --verify debe pasar.

    Solo corre si los parquets reservados estan presentes (local). En
    CI no existen (gitignored).
    """
    rc = hm.main(["--verify"])
    assert rc == 0, "manifest no coincide con parquets (regenerar con --update)"


# --- H-03: interseccion ---

def test_intersection_existe_y_estructura():
    p = ROOT / "data" / "mappings" / "holdout_intersection.json"
    assert p.exists(), "holdout_intersection.json no generado"
    d = json.loads(p.read_text(encoding="utf-8"))
    for k in (
        "schema_version", "n_cohort", "n_r3000", "n_intersection",
        "n_only_cohort", "n_only_r3000", "cohort_subset_of_r3000",
        "intersection_sample", "only_cohort_sample", "only_r3000_sample",
    ):
        assert k in d, "falta " + k
    assert d["schema_version"] == 1
    assert d["n_cohort"] == 268
    assert d["n_r3000"] == 3423


def test_intersection_valores_reales():
    """Confirmado con datos reales 2026-10-28: interseccion=266, solo_cohort=2.

    El auditor aviso (H-03): no asumir universos disjuntos. Con 266
    de 268 tickers compartidos, los resultados de cohort y r3000 NO
    son dos confirmaciones independientes para esos 266.
    """
    p = ROOT / "data" / "mappings" / "holdout_intersection.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["n_intersection"] == 266, "cambio la interseccion, regenerar"
    assert d["n_only_cohort"] == 2, "cambio solo_cohort, regenerar"
    assert d["cohort_subset_of_r3000"] is False, (
        "cohort NO es subset estricto de r3000 (2 tickers fuera)"
    )
    assert d["n_intersection"] + d["n_only_cohort"] == d["n_cohort"]
    assert d["n_intersection"] + d["n_only_r3000"] == d["n_r3000"]


def test_intersection_suma_coherente():
    p = ROOT / "data" / "mappings" / "holdout_intersection.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["n_intersection"] + d["n_only_cohort"] == d["n_cohort"]
    assert d["n_intersection"] + d["n_only_r3000"] == d["n_r3000"]


def test_intersection_nota_auditor():
    """La nota deja claro que compartir observaciones invalida la
    independencia para esos tickers."""
    p = ROOT / "data" / "mappings" / "holdout_intersection.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    nota = d.get("note", "").lower()
    assert "no son dos confirmaciones independientes" in nota
    assert "comparten" in nota


# --- Integracion: script completo ---

@_skip_sin_parquets
def test_intersection_main_devuelve_0(tmp_path, monkeypatch):
    """main() ejecuta sin error. Solo con parquets presentes (local)."""
    rc = hi.main()
    assert rc == 0