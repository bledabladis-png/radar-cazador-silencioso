# -*- coding: utf-8 -*-
"""Tests de sec_13f.manifest (FA-1.3). Sin red."""
import json

import pytest

from src.institutional_accumulation.sec_13f import manifest


def _touch(p, content=b"data"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def _make_parquet_files(base, names=("SUBMISSION", "COVERPAGE")):
    """Crea ficheros fake en base/ para que build_manifest no falle."""
    files = {}
    for name in names:
        p = base / (name + ".parquet")
        _touch(p, content=b"fake_" + name.encode())
        files[name] = p
    return files


def _make_tsv_files(base, names=("SUBMISSION", "COVERPAGE")):
    files = {}
    for name in names:
        p = base / (name + ".tsv")
        _touch(p, content=b"a\tb\n")
        files[name] = p
    return files

def test_build_manifest_estructura(tmp_path):
    pq = _make_parquet_files(tmp_path)
    tsv = _make_tsv_files(tmp_path)
    m = manifest.build_manifest(
        quarter="2026Q1",
        source_period="01mar2026-31may2026",
        source_url="https://example/x.zip",
        source_filename="x.zip",
        source_sha256="a" * 64,
        tsv_files=tsv,
        parquet_files=pq,
        row_counts={"SUBMISSION": 100, "COVERPAGE": 100},
        validation={"expected_files_present": True, "zip_crc_valid": True, "schema_valid": True},
        project_root=tmp_path,
    )
    assert m["dataset"] == "SEC_FORM_13F"
    assert m["quarter"] == "2026Q1"
    assert m["source_period"] == "01mar2026-31may2026"
    assert m["schema_version"]
    assert m["parser_version"]
    assert m["generated_at_utc"]


def test_build_manifest_tres_niveles_hash(tmp_path):
    pq = _make_parquet_files(tmp_path)
    tsv = _make_tsv_files(tmp_path)
    m = manifest.build_manifest(
        quarter="2026Q1",
        source_period="01mar2026-31may2026",
        source_url="https://example/x.zip",
        source_filename="x.zip",
        source_sha256="a" * 64,
        tsv_files=tsv,
        parquet_files=pq,
        row_counts={"SUBMISSION": 100, "COVERPAGE": 100},
        validation={"expected_files_present": True, "zip_crc_valid": True, "schema_valid": True},
        project_root=tmp_path,
    )
    assert m["source"]["sha256"] == "a" * 64
    assert "SUBMISSION" in m["tsv_files"]
    assert m["tsv_files"]["SUBMISSION"]["sha256"]
    assert m["tsv_files"]["SUBMISSION"]["rows"] == 100
    assert "SUBMISSION" in m["parquet_files"]
    assert m["parquet_files"]["SUBMISSION"]["sha256"]
    assert m["parquet_files"]["SUBMISSION"]["rows"] == 100


def test_build_manifest_validation_flags(tmp_path):
    pq = _make_parquet_files(tmp_path)
    tsv = _make_tsv_files(tmp_path)
    m = manifest.build_manifest(
        quarter="2026Q1",
        source_period="01mar2026-31may2026",
        source_url="x",
        source_filename="x",
        source_sha256="a" * 64,
        tsv_files=tsv,
        parquet_files=pq,
        row_counts={},
        validation={},
        project_root=tmp_path,
    )
    assert m["validation"]["expected_files_present"] is False
    assert m["validation"]["zip_crc_valid"] is False
    assert m["validation"]["schema_valid"] is False

def test_build_manifest_rutas_relativas(tmp_path):
    pq = _make_parquet_files(tmp_path)
    tsv = _make_tsv_files(tmp_path)
    m = manifest.build_manifest(
        quarter="2026Q1",
        source_period="x",
        source_url="u",
        source_filename="f",
        source_sha256="a" * 64,
        tsv_files=tsv,
        parquet_files=pq,
        row_counts={},
        validation={},
        project_root=tmp_path,
    )
    for name in pq:
        file_field = m["parquet_files"][name]["file"]
        assert not file_field.startswith("/")
        assert not ("D:" in file_field or "C:" in file_field)


def test_build_manifest_falla_si_parquet_no_existe(tmp_path):
    pq = {"SUBMISSION": tmp_path / "no_existe.parquet"}
    tsv = _make_tsv_files(tmp_path)
    with pytest.raises(FileNotFoundError, match="Parquet no existe"):
        manifest.build_manifest(
            quarter="2026Q1", source_period="x",
            source_url="u", source_filename="f", source_sha256="a" * 64,
            tsv_files=tsv, parquet_files=pq,
            row_counts={}, validation={}, project_root=tmp_path,
        )


def test_build_manifest_falla_si_tsv_no_existe(tmp_path):
    pq = _make_parquet_files(tmp_path)
    tsv = {"SUBMISSION": tmp_path / "no_existe.tsv"}
    with pytest.raises(FileNotFoundError, match="TSV no existe"):
        manifest.build_manifest(
            quarter="2026Q1", source_period="x",
            source_url="u", source_filename="f", source_sha256="a" * 64,
            tsv_files=tsv, parquet_files=pq,
            row_counts={}, validation={}, project_root=tmp_path,
        )

def test_write_manifest_crea_dirs(tmp_path):
    m = {"dataset": "x", "quarter": "2026Q1"}
    path = manifest.write_manifest(m, tmp_path / "manifests" / "sec_13f_2026Q1.json")
    assert path.exists()


def test_write_manifest_atomico_sin_tmp(tmp_path):
    m = {"dataset": "x"}
    manifest.write_manifest(m, tmp_path / "m.json")
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_write_manifest_json_valido(tmp_path):
    m = {"dataset": "x", "n": 42, "s": "abc"}
    p = manifest.write_manifest(m, tmp_path / "m.json")
    with open(p, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["dataset"] == "x"
    assert loaded["n"] == 42
    assert loaded["s"] == "abc"


def test_read_manifest_roundtrip(tmp_path):
    m = {"dataset": "x", "quarter": "2026Q1"}
    p = manifest.write_manifest(m, tmp_path / "m.json")
    loaded = manifest.read_manifest(p)
    assert loaded == m


def test_default_project_root_apunta_a_raiz_repo():
    """P18/P26: el default no es literal; apunta a la raiz del repo."""
    from src.institutional_accumulation.sec_13f import manifest as m

    root = m.DEFAULT_PROJECT_ROOT
    expected_subpath = root / 'src' / 'institutional_accumulation'
    assert expected_subpath.exists(), (
        f'DEFAULT_PROJECT_ROOT ({root}) no contiene src/institutional_accumulation'
    )


def test_default_project_root_no_es_literal_hardcoded():
    """P18/P26: el codigo fuente no contiene path absoluto hardcoded."""
    import inspect
    from src.institutional_accumulation.sec_13f import manifest as m

    src = inspect.getsource(m)
    # Prohibido D:\Macro_Sectorial o cualquier r'D:\' literal
    assert 'D:\\\\Macro_Sectorial' not in src
    assert 'r"D:' not in src
    assert "r'D:" not in src


def test_read_manifest_no_existe(tmp_path):
    with pytest.raises(FileNotFoundError, match="Manifest no existe"):
        manifest.read_manifest(tmp_path / "no_existe.json")