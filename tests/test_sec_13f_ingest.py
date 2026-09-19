# -*- coding: utf-8 -*-
"""Tests de sec_13f.ingest (FA-1.4). Sin red (mock de descarga)."""
import zipfile
from pathlib import Path
from unittest.mock import patch

from src.institutional_accumulation.sec_13f import ingest
from src.institutional_accumulation.sec_13f.schema import (
    EXPECTED_COLUMNS,
    EXPECTED_FILES,
)


def _make_tsv_content(tsv_name):
    """Genera contenido minimo valido para un TSV."""
    cols = EXPECTED_COLUMNS[tsv_name]
    header = "\t".join(cols)
    row = []
    for c in cols:
        if "DATE" in c or c == "PERIODOFREPORT":
            row.append("31-MAR-2026")
        else:
            row.append("0")
    return header + "\n" + "\t".join(row) + "\n"


def _make_synthetic_zip(zip_path):
    """Crea ZIP con los 7 TSVs esperados."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in EXPECTED_FILES:
            name = f.replace(".tsv", "")
            content = _make_tsv_content(name).encode("utf-8")
            zf.writestr(f, content)


def _fake_download(period, dest_dir, **kwargs):
    """Sustituye a download_13f_zip. Crea el ZIP sintetico."""
    zip_path = Path(dest_dir) / (period + "_form13f.zip")
    _make_synthetic_zip(zip_path)
    return zip_path

def test_ingest_end_to_end_estructura(tmp_path):
    with patch("src.institutional_accumulation.sec_13f.ingest.download_13f_zip",
               side_effect=_fake_download):
        result = ingest.ingest_13f(
            "2026Q1", "01mar2026-31may2026",
            base_dir=tmp_path, project_root=tmp_path,
        )
    assert result["quarter"] == "2026Q1"
    assert result["source_period"] == "01mar2026-31may2026"
    assert len(result["tsv_paths"]) == 7
    assert len(result["parquet_paths"]) == 7
    assert len(result["row_counts"]) == 7
    for key in ["SUBMISSION", "COVERPAGE", "SUMMARYPAGE", "OTHERMANAGER",
                "OTHERMANAGER2", "SIGNATURE", "INFOTABLE"]:
        assert key in result["tsv_paths"]
        assert key in result["parquet_paths"]
        assert key in result["row_counts"]
        assert result["row_counts"][key] == 1


def test_ingest_crea_7_parquets(tmp_path):
    with patch("src.institutional_accumulation.sec_13f.ingest.download_13f_zip",
               side_effect=_fake_download):
        ingest.ingest_13f(
            "2026Q1", "01mar2026-31may2026",
            base_dir=tmp_path, project_root=tmp_path,
        )
    processed_dir = tmp_path / "processed" / "2026Q1"
    for name in ["SUBMISSION", "COVERPAGE", "SUMMARYPAGE", "OTHERMANAGER",
                 "OTHERMANAGER2", "SIGNATURE", "INFOTABLE"]:
        p = processed_dir / (name + ".parquet")
        assert p.exists()
        assert p.stat().st_size > 0


def test_ingest_crea_manifest_en_disco(tmp_path):
    with patch("src.institutional_accumulation.sec_13f.ingest.download_13f_zip",
               side_effect=_fake_download):
        result = ingest.ingest_13f(
            "2026Q1", "01mar2026-31may2026",
            base_dir=tmp_path, project_root=tmp_path,
        )
    manifest_path = tmp_path / "manifests" / "sec_13f_2026Q1.json"
    assert manifest_path.exists()
    assert result["manifest_path"] == manifest_path

def test_ingest_manifest_3_niveles_hash(tmp_path):
    with patch("src.institutional_accumulation.sec_13f.ingest.download_13f_zip",
               side_effect=_fake_download):
        result = ingest.ingest_13f(
            "2026Q1", "01mar2026-31may2026",
            base_dir=tmp_path, project_root=tmp_path,
        )
    m = result["manifest"]
    # Nivel 1: hash del ZIP
    assert m["source"]["sha256"]
    assert len(m["source"]["sha256"]) == 64
    # Nivel 2: hash de los TSVs
    assert len(m["tsv_files"]) == 7
    for name, meta in m["tsv_files"].items():
        assert meta["sha256"]
        assert meta["size_bytes"] > 0
        assert meta["rows"] == 1
    # Nivel 3: hash de los parquets
    assert len(m["parquet_files"]) == 7
    for name, meta in m["parquet_files"].items():
        assert meta["sha256"]
        assert meta["size_bytes"] > 0
        assert meta["rows"] == 1


def test_ingest_manifest_validation_flags(tmp_path):
    with patch("src.institutional_accumulation.sec_13f.ingest.download_13f_zip",
               side_effect=_fake_download):
        result = ingest.ingest_13f(
            "2026Q1", "01mar2026-31may2026",
            base_dir=tmp_path, project_root=tmp_path,
        )
    v = result["manifest"]["validation"]
    assert v["expected_files_present"] is True
    assert v["zip_crc_valid"] is True
    assert v["schema_valid"] is True


def test_ingest_manifest_quarter_y_period_separados(tmp_path):
    with patch("src.institutional_accumulation.sec_13f.ingest.download_13f_zip",
               side_effect=_fake_download):
        result = ingest.ingest_13f(
            "2026Q1", "01mar2026-31may2026",
            base_dir=tmp_path, project_root=tmp_path,
        )
    m = result["manifest"]
    assert m["quarter"] == "2026Q1"
    assert m["source_period"] == "01mar2026-31may2026"


def test_ingest_segunda_llamada_reutiliza_cache(tmp_path):
    with patch("src.institutional_accumulation.sec_13f.ingest.download_13f_zip",
               side_effect=_fake_download) as mock_dl:
        result1 = ingest.ingest_13f(
            "2026Q1", "01mar2026-31may2026",
            base_dir=tmp_path, project_root=tmp_path,
        )
        result2 = ingest.ingest_13f(
            "2026Q1", "01mar2026-31may2026",
            base_dir=tmp_path, project_root=tmp_path,
        )
    assert mock_dl.call_count == 2
    # Hashes identicos (idempotencia de parquet snappy sobre mismo df)
    for name in result1["parquet_paths"]:
        assert (result1["manifest"]["parquet_files"][name]["sha256"]
                == result2["manifest"]["parquet_files"][name]["sha256"])


def test_ingest_estructura_directorios(tmp_path):
    with patch("src.institutional_accumulation.sec_13f.ingest.download_13f_zip",
               side_effect=_fake_download):
        ingest.ingest_13f(
            "2026Q1", "01mar2026-31may2026",
            base_dir=tmp_path, project_root=tmp_path,
        )
    assert (tmp_path / "raw" / "01mar2026-31may2026").exists()
    assert (tmp_path / "raw" / "01mar2026-31may2026" / "extracted").exists()
    assert (tmp_path / "processed" / "2026Q1").exists()
    assert (tmp_path / "manifests").exists()