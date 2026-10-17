# -*- coding: utf-8 -*-
"""Tests H23 (dictamen externo 2026-10-17): futures_status debe
publicar UNAVAILABLE si el metadato FUTURE_SETTLEMENT falta o no
trae status. Nunca inventar OK ni BLOCKED por ausencia.

El commit 05a86817 sustituyo el literal 'BLOCKED' por la lectura de
temporal_meta.FUTURE_SETTLEMENT.status. El auditor exigio blindar el
caso de metadato ausente o invalido, no solo el camino feliz.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "tests" / "fixtures"


def _load_inputs():
    df = pd.read_parquet(FIXTURES / "mte_input_df.parquet")
    fs = pd.read_csv(
        FIXTURES / "mte_financial_score.csv",
        index_col=0, parse_dates=True,
    ).iloc[:, 0]
    pcr = json.loads(
        (FIXTURES / "mte_pcr_data.json").read_text(encoding="utf-8")
    )
    darkpool = json.loads(
        (FIXTURES / "mte_darkpool_data.json").read_text(encoding="utf-8")
    )
    return df, fs, pcr, darkpool


def _patch_state(monkeypatch, path):
    import indicators.mte.state as mte_state
    import indicators.mte.engine as mte_engine
    monkeypatch.setattr(mte_state, "MTE_STATE_FILE", str(path))
    monkeypatch.setattr(mte_engine._mte_state, "MTE_STATE_FILE", str(path))


def _run_compute_mte(tmp_path, monkeypatch, temporal_meta):
    state_path = tmp_path / "mte_state.json"
    prev = json.loads(
        (FIXTURES / "mte_previous_state.json").read_text(encoding="utf-8")
    )
    state_path.write_text(json.dumps(prev), encoding="utf-8")
    _patch_state(monkeypatch, state_path)

    df, fs, pcr, darkpool = _load_inputs()
    import indicators.mte.engine as eng
    result = eng.compute_mte(
        df, fs, None, None, pcr_data=pcr, darkpool_data=darkpool,
        temporal_meta=temporal_meta,
    )
    assert result is not None
    return json.loads(state_path.read_text("utf-8"))


def test_futures_status_ok_cuando_metadato_lo_dice(tmp_path, monkeypatch):
    """Metadato con status=OK -> state.futures_status=OK."""
    tmeta = {"by_contract": {"FUTURE_SETTLEMENT": {"status": "OK"}}}
    saved = _run_compute_mte(tmp_path, monkeypatch, tmeta)
    assert saved["futures_status"] == "OK"


def test_futures_status_blocked_cuando_metadato_lo_dice(tmp_path, monkeypatch):
    """Metadato con status=BLOCKED -> state.futures_status=BLOCKED."""
    tmeta = {"by_contract": {"FUTURE_SETTLEMENT": {"status": "BLOCKED"}}}
    saved = _run_compute_mte(tmp_path, monkeypatch, tmeta)
    assert saved["futures_status"] == "BLOCKED"


def test_futures_status_unavailable_si_falta_contrato(tmp_path, monkeypatch):
    """by_contract sin FUTURE_SETTLEMENT -> UNAVAILABLE (no None, no OK)."""
    tmeta = {"by_contract": {}}
    saved = _run_compute_mte(tmp_path, monkeypatch, tmeta)
    assert saved["futures_status"] == "UNAVAILABLE"


def test_futures_status_unavailable_si_contrato_sin_status(
        tmp_path, monkeypatch):
    """FUTURE_SETTLEMENT sin status -> UNAVAILABLE."""
    tmeta = {"by_contract": {"FUTURE_SETTLEMENT": {}}}
    saved = _run_compute_mte(tmp_path, monkeypatch, tmeta)
    assert saved["futures_status"] == "UNAVAILABLE"


def test_futures_status_unavailable_si_tmeta_none(tmp_path, monkeypatch):
    """temporal_meta=None -> UNAVAILABLE."""
    saved = _run_compute_mte(tmp_path, monkeypatch, None)
    assert saved["futures_status"] == "UNAVAILABLE"


def test_futures_status_unavailable_si_status_vacio(tmp_path, monkeypatch):
    """status="" (string vacio) -> UNAVAILABLE (no se acepta vacio)."""
    tmeta = {"by_contract": {"FUTURE_SETTLEMENT": {"status": ""}}}
    saved = _run_compute_mte(tmp_path, monkeypatch, tmeta)
    assert saved["futures_status"] == "UNAVAILABLE"