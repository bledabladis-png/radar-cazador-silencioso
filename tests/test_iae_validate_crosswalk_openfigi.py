"""Tests del orquestador scripts/iae_validate_crosswalk_openfigi.py.

No tocan red: monkeypatch de map_identifiers + git_head + MAPPINGS/OUT_DIR.
Verifican los artefactos de evidencia externa exigidos por el dictamen v4
(punto 7): script_version en _input.json, _summary.txt y HASHES_*.txt.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (Path(__file__).resolve().parent.parent
               / "scripts" / "iae_validate_crosswalk_openfigi.py")


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "iae_validate_crosswalk_openfigi", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def env(tmp_path, monkeypatch):
    mod = _load_module()
    mappings = tmp_path / "mappings"
    mappings.mkdir()
    out_dir = mappings / "openfigi_requeries"
    (mappings / "cusip_to_radar_figi.csv").write_text(
        "cusip,radar_ticker,share_class_figi\n"
        "037833100,AAPL,BBG001S5N8V8\n"
        "02079K107,GOOG,BBG009S3NB21\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "MAPPINGS", mappings)
    monkeypatch.setattr(mod, "OUT_DIR", out_dir)
    monkeypatch.setattr(mod, "git_head", lambda: "FAKEHEAD")
    monkeypatch.setattr(mod.time, "sleep", lambda _x: None)
    fake_data = {
        "037833100": {"ok": True, "error": None,
                      "data": [{"shareClassFIGI": "BBG001S5N8V8",
                                "ticker": "AAPL", "exchCode": "US",
                                "name": "APPLE INC"}]},
        "02079K107": {"ok": True, "error": None,
                      "data": [{"shareClassFIGI": "BBG009S3NB21",
                                "ticker": "GOOG", "exchCode": "US",
                                "name": "ALPHABET INC-CL C"}]},
    }

    def fake_map(id_type, values, exch_code=None, api_key=None):
        return {v: fake_data[v] for v in values}

    monkeypatch.setattr(mod, "map_identifiers", fake_map)
    return mod, out_dir


def _one(dirpath, pattern):
    hits = list(Path(dirpath).glob(pattern))
    assert len(hits) == 1, f"se esperaba 1 match para {pattern}, hay {len(hits)}"
    return hits[0]


def test_artefactos_generados(env):
    mod, out_dir = env
    rc = mod.main()
    assert rc == 0
    assert _one(out_dir, "*_input.json").exists()
    assert _one(out_dir, "*_raw.json").exists()
    assert _one(out_dir, "*_summary.txt").exists()
    assert _one(out_dir, "HASHES_*.txt").exists()


def test_script_version_en_input_json(env):
    mod, out_dir = env
    mod.main()
    inp = json.loads(_one(out_dir, "*_input.json").read_text(encoding="utf-8"))
    assert inp["script_version"] == mod.SCRIPT_VERSION


def test_script_version_en_summary(env):
    mod, out_dir = env
    mod.main()
    txt = _one(out_dir, "*_summary.txt").read_text(encoding="utf-8")
    assert "Script version: " + mod.SCRIPT_VERSION in txt


def test_script_version_en_hashes_header(env):
    mod, out_dir = env
    mod.main()
    txt = _one(out_dir, "HASHES_*.txt").read_text(encoding="utf-8")
    assert "# Script version: " + mod.SCRIPT_VERSION in txt


def test_exit_code_fail_cuando_mismatch(env, monkeypatch):
    mod, out_dir = env

    def fake_map_mismatch(id_type, values, exch_code=None, api_key=None):
        return {v: {"ok": True, "error": None,
                    "data": [{"shareClassFIGI": "BBG_MISMATCH",
                              "ticker": "X", "exchCode": "US",
                              "name": "X"}]} for v in values}

    monkeypatch.setattr(mod, "map_identifiers", fake_map_mismatch)
    rc = mod.main()
    assert rc == 1