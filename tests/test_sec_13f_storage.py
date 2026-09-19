# -*- coding: utf-8 -*-
"""Tests de sec_13f.storage (FA-1.3). Sin red."""

import pandas as pd
import pytest

from src.institutional_accumulation.sec_13f import storage
from src.institutional_accumulation.sec_13f.schema import EXPECTED_FILES


def _make_all_7_dfs():
    """Devuelve dict {nombre: DataFrame simple} para los 7 TSVs."""
    dfs = {}
    for filename in EXPECTED_FILES:
        name = filename.replace(".tsv", "")
        dfs[name] = pd.DataFrame({"col": [1, 2, 3]})
    return dfs

def test_write_parquets_devuelve_metadata(tmp_path):
    dfs = {"SUBMISSION": pd.DataFrame({"col": [1, 2, 3]})}
    result = storage.write_parquets(dfs, "2026Q1", base_dir=tmp_path)
    assert "SUBMISSION" in result
    meta = result["SUBMISSION"]
    assert meta["path"].exists()
    assert meta["rows"] == 3
    assert isinstance(meta["sha256"], str)
    assert len(meta["sha256"]) == 64
    assert meta["size_bytes"] > 0


def test_write_parquets_dict_vacio_lanza(tmp_path):
    with pytest.raises(ValueError, match="no vacio"):
        storage.write_parquets({}, "2026Q1", base_dir=tmp_path)


def test_write_parquets_no_dataframe_lanza(tmp_path):
    with pytest.raises(TypeError, match="no es DataFrame"):
        storage.write_parquets({"X": [1, 2, 3]}, "2026Q1", base_dir=tmp_path)


def test_write_parquets_crea_directorios(tmp_path):
    dfs = {"SUBMISSION": pd.DataFrame({"col": [1]})}
    storage.write_parquets(dfs, "2026Q1", base_dir=tmp_path)
    expected = tmp_path / "processed" / "2026Q1" / "SUBMISSION.parquet"
    assert expected.exists()


def test_write_parquets_no_deja_tmp(tmp_path):
    dfs = {"SUBMISSION": pd.DataFrame({"col": [1]})}
    storage.write_parquets(dfs, "2026Q1", base_dir=tmp_path)
    tmp_files = list((tmp_path / "processed" / "2026Q1").glob("*.tmp"))
    assert tmp_files == []


def test_write_parquets_idempotente(tmp_path):
    dfs = {"SUBMISSION": pd.DataFrame({"col": [1, 2, 3]})}
    r1 = storage.write_parquets(dfs, "2026Q1", base_dir=tmp_path)
    r2 = storage.write_parquets(dfs, "2026Q1", base_dir=tmp_path)
    assert r1["SUBMISSION"]["sha256"] == r2["SUBMISSION"]["sha256"]

def test_load_parquets_roundtrip(tmp_path):
    dfs = _make_all_7_dfs()
    storage.write_parquets(dfs, "2026Q1", base_dir=tmp_path)
    loaded = storage.load_parquets("2026Q1", base_dir=tmp_path)
    assert len(loaded) == 7
    for name in dfs:
        assert name in loaded
        assert len(loaded[name]) == 3


def test_load_parquets_dir_no_existe(tmp_path):
    with pytest.raises(FileNotFoundError, match="processed no existe"):
        storage.load_parquets("2026Q1", base_dir=tmp_path)


def test_load_parquets_falta_uno(tmp_path):
    dfs = _make_all_7_dfs()
    storage.write_parquets(dfs, "2026Q1", base_dir=tmp_path)
    (tmp_path / "processed" / "2026Q1" / "INFOTABLE.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="Parquet faltante"):
        storage.load_parquets("2026Q1", base_dir=tmp_path)


def test_get_manifest_path(tmp_path):
    p = storage.get_manifest_path("2026Q1", base_dir=tmp_path)
    assert p.name == "sec_13f_2026Q1.json"
    assert p.parent.name == "manifests"