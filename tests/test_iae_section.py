"""Tests unitarios de src/pipeline/iae_section (E.3.b).

Cubren:
  - _list_available_quarters: deteccion de trimestres.
  - compute_iae_section: paths STALE / OK / ERROR.
  - Estructura del dict devuelto.
"""
from __future__ import annotations

from unittest.mock import patch

from src.pipeline import iae_section


# --- _list_available_quarters -------------------------------------------

def test_list_quarters_dir_no_existe(tmp_path):
    fake_dir = tmp_path / "nonexistent"
    with patch.object(iae_section, "DATA_DIR", fake_dir):
        assert iae_section._list_available_quarters() == []


def test_list_quarters_vacio(tmp_path):
    d = tmp_path / "sec_13f" / "processed"
    d.mkdir(parents=True)
    with patch.object(iae_section, "DATA_DIR", d):
        assert iae_section._list_available_quarters() == []


def test_list_quarters_ignora_folders_sin_infotable(tmp_path):
    d = tmp_path / "sec_13f" / "processed"
    (d / "2025Q4").mkdir(parents=True)
    (d / "2026Q1").mkdir(parents=True)
    (d / "2026Q1" / "INFOTABLE.parquet").write_bytes(b"")
    with patch.object(iae_section, "DATA_DIR", d):
        out = iae_section._list_available_quarters()
        assert out == [("2026Q1", "2026-03-31")]


def test_list_quarters_ordena_cronologicamente(tmp_path):
    d = tmp_path / "sec_13f" / "processed"
    for f in ("2026Q1", "2025Q4", "2025Q3", "2026Q2"):
        (d / f).mkdir(parents=True)
        (d / f / "INFOTABLE.parquet").write_bytes(b"")
    with patch.object(iae_section, "DATA_DIR", d):
        out = iae_section._list_available_quarters()
        names = [x[0] for x in out]
        assert names == ["2025Q3", "2025Q4", "2026Q1", "2026Q2"]


def test_list_quarters_ignora_carpetas_invalidas(tmp_path):
    d = tmp_path / "sec_13f" / "processed"
    (d / "2025Q4").mkdir(parents=True)
    (d / "2025Q4" / "INFOTABLE.parquet").write_bytes(b"")
    (d / "foo").mkdir()
    (d / "foo" / "INFOTABLE.parquet").write_bytes(b"")
    (d / "2024Q9").mkdir()  # trimestre invalido
    (d / "2024Q9" / "INFOTABLE.parquet").write_bytes(b"")
    with patch.object(iae_section, "DATA_DIR", d):
        out = iae_section._list_available_quarters()
        assert out == [("2025Q4", "2025-12-31")]

# --- compute_iae_section: STALE ----------------------------------------

def _stale_env(tmp_path, n_quarters):
    d = tmp_path / "sec_13f" / "processed"
    for f in ["2025Q1", "2025Q2", "2025Q3", "2025Q4"][:n_quarters]:
        (d / f).mkdir(parents=True)
        (d / f / "INFOTABLE.parquet").write_bytes(b"")
    return d


def test_iae_section_stale_sin_quarters(tmp_path):
    d = _stale_env(tmp_path, 0)
    with patch.object(iae_section, "DATA_DIR", d):
        r = iae_section.compute_iae_section(None, "TEST")
    assert r["status"] == "STALE"
    assert r["period_current"] is None
    assert r["nipc_total"] is None


def test_iae_section_stale_un_solo_quarter(tmp_path):
    d = _stale_env(tmp_path, 1)
    with patch.object(iae_section, "DATA_DIR", d):
        r = iae_section.compute_iae_section(None, "TEST")
    assert r["status"] == "STALE"
    assert r["period_previous"] == "2025Q1"
    assert r["period_current"] is None


# --- compute_iae_section: OK -------------------------------------------

def _ok_env(tmp_path):
    return _stale_env(tmp_path, 4)


def _fake_result():
    return {
        "nipc_total": -100.0,
        "nipc_sole":  -200.0,
        "nipc_dfnd":   150.0,
        "nipc_otr":    -50.0,
        "n_both": 10, "n_new": 3, "n_exit": 2,
        "n_delta_observable": 15,
        "n_unresolved_identity": 0,
        "coverage_status": "VALID",
        "coverage_quality": "COMPLETE",
        "evidence_class": "CONTRACTUAL",
        "target_q4_size": 240,
        "target_q1_size": 240,
    }


def test_iae_section_ok(tmp_path):
    d = _ok_env(tmp_path)
    with patch.object(iae_section, "DATA_DIR", d), \
         patch("src.institutional_accumulation.pipeline_contractual"
               ".run_contractual_nipc", return_value=_fake_result()):
        r = iae_section.compute_iae_section(None, "TEST")
    assert r["status"] == "OK"
    assert r["period_previous"] == "2025Q3"
    assert r["period_current"] == "2025Q4"
    assert r["nipc_total"] == -100.0
    assert r["n_delta_observable"] == 15
    assert r["coverage_status"] == "VALID"
    assert r["evidence_class"] == "CONTRACTUAL"


def test_iae_section_ok_cobertura_catalogo(tmp_path):
    d = _ok_env(tmp_path)
    fake = _fake_result()
    fake["target_q4_size"] = 242
    fake["target_q1_size"] = 242
    with patch.object(iae_section, "DATA_DIR", d), \
         patch("src.institutional_accumulation.pipeline_contractual"
               ".run_contractual_nipc", return_value=fake):
        r = iae_section.compute_iae_section(None, "TEST")
    assert r["catalog_keys_total"] == 242
    assert r["catalog_keys_observed"] == 242
    assert abs(r["catalog_coverage_declared"] - 1.0) < 1e-9


# --- compute_iae_section: ERROR ----------------------------------------

def test_iae_section_error(tmp_path):
    d = _ok_env(tmp_path)
    with patch.object(iae_section, "DATA_DIR", d), \
         patch("src.institutional_accumulation.pipeline_contractual"
               ".run_contractual_nipc",
               side_effect=RuntimeError("boom")):
        r = iae_section.compute_iae_section(None, "TEST")
    assert r["status"] == "ERROR"
    assert "RuntimeError" in r["error"]
    assert "boom" in r["error"]
    assert r["nipc_total"] is None


# --- estructura del dict -----------------------------------------------

ALL_KEYS = (
    "status","period_previous","period_current",
    "iso_previous","iso_current",
    "nipc_total","nipc_sole","nipc_dfnd","nipc_otr",
    "n_delta_observable","n_both","n_new","n_exit","n_unresolved_identity",
    "coverage_status","coverage_quality",
    "catalog_coverage_declared","catalog_keys_total","catalog_keys_observed",
    "evidence_class","stale_reason","catalog_coverage_warning","error",
)


def test_iae_section_todas_las_claves_presentes_stale(tmp_path):
    d = _stale_env(tmp_path, 0)
    with patch.object(iae_section, "DATA_DIR", d):
        r = iae_section.compute_iae_section(None, "TEST")
    for k in ALL_KEYS:
        assert k in r, f"falta {k} en STALE"


def test_iae_section_todas_las_claves_presentes_ok(tmp_path):
    d = _ok_env(tmp_path)
    with patch.object(iae_section, "DATA_DIR", d), \
         patch("src.institutional_accumulation.pipeline_contractual"
               ".run_contractual_nipc", return_value=_fake_result()):
        r = iae_section.compute_iae_section(None, "TEST")
    for k in ALL_KEYS:
        assert k in r, f"falta {k} en OK"


# --- stale_reason (Bug B: Official List pendiente) ----------------------

def test_iae_section_stale_insufficient_quarters_reason(tmp_path):
    d = _stale_env(tmp_path, 0)
    with patch.object(iae_section, "DATA_DIR", d):
        r = iae_section.compute_iae_section(None, "TEST")
    assert r["status"] == "STALE"
    assert r["stale_reason"] == "insufficient_quarters"


def test_iae_section_official_list_pending(tmp_path):
    d = _ok_env(tmp_path)
    with patch.object(iae_section, "DATA_DIR", d), \
         patch("src.institutional_accumulation.pipeline_contractual"
               ".run_contractual_nipc",
               side_effect=FileNotFoundError(
                   "Official List no existe: /tmp/13flist_2025Q4.txt")):
        r = iae_section.compute_iae_section(None, "TEST")
    assert r["status"] == "STALE"
    assert r["stale_reason"] == "official_list_pending"
    assert r["period_current"] == "2025Q4"
    assert r["nipc_total"] is None
    assert "Official List" in r["error"]


def test_iae_section_ok_stale_reason_none(tmp_path):
    d = _ok_env(tmp_path)
    with patch.object(iae_section, "DATA_DIR", d), \
         patch("src.institutional_accumulation.pipeline_contractual"
               ".run_contractual_nipc", return_value=_fake_result()):
        r = iae_section.compute_iae_section(None, "TEST")
    assert r["status"] == "OK"
    assert r["stale_reason"] is None


# --- catalog_coverage_warning (Fase 3.1) --------------------------------

def test_iae_section_coverage_warning_true_si_bajo_umbral(tmp_path):
    """Con cobertura < 90%, warning=True."""
    d = _ok_env(tmp_path)
    fake = _fake_result()
    # 220/242 = 90.9% -> NO warning. Bajamos a 200/242 = 82.6% -> SI warning.
    fake["target_q4_size"] = 200
    fake["target_q1_size"] = 200
    with patch.object(iae_section, "DATA_DIR", d), \
         patch("src.institutional_accumulation.pipeline_contractual"
               ".run_contractual_nipc", return_value=fake):
        r = iae_section.compute_iae_section(None, "TEST")
    assert r["status"] == "OK"
    assert r["catalog_coverage_warning"] is True


def test_iae_section_coverage_warning_false_si_sobre_umbral(tmp_path):
    """Con cobertura >= 90%, warning=False."""
    d = _ok_env(tmp_path)
    fake = _fake_result()
    fake["target_q4_size"] = 240
    fake["target_q1_size"] = 240
    with patch.object(iae_section, "DATA_DIR", d), \
         patch("src.institutional_accumulation.pipeline_contractual"
               ".run_contractual_nipc", return_value=fake):
        r = iae_section.compute_iae_section(None, "TEST")
    assert r["status"] == "OK"
    assert r["catalog_coverage_warning"] is False


def test_iae_section_coverage_warning_none_en_stale(tmp_path):
    """En STALE, warning=None (no aplica)."""
    d = _stale_env(tmp_path, 0)
    with patch.object(iae_section, "DATA_DIR", d):
        r = iae_section.compute_iae_section(None, "TEST")
    assert r["status"] == "STALE"
    assert r["catalog_coverage_warning"] is None


def test_iae_section_coverage_warning_none_en_error(tmp_path):
    """En ERROR, warning=None (no aplica)."""
    d = _ok_env(tmp_path)
    with patch.object(iae_section, "DATA_DIR", d), \
         patch("src.institutional_accumulation.pipeline_contractual"
               ".run_contractual_nipc", side_effect=RuntimeError("boom")):
        r = iae_section.compute_iae_section(None, "TEST")
    assert r["status"] == "ERROR"
    assert r["catalog_coverage_warning"] is None
