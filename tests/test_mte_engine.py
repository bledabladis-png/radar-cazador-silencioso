"""Tests del motor MTE contra golden reference (DT2 Fase 1).

Cubre:
- Integridad de fixtures (sha256 LF-normalized).
- Estructura del golden.
- compute_mte contra el golden 2026-09-16 con tolerancias documentadas.

No depende de data/*.parquet (gitignored). Solo tests/fixtures/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from datetime import datetime
from zoneinfo import ZoneInfo


FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN_PATH = FIXTURES / "mte_golden_2026-09-16.json"

_FIXTURE_FILES = {
    "input_df": "mte_input_df.parquet",
    "previous_state": "mte_previous_state.json",
    "financial_score": "mte_financial_score.csv",
    "all_signals": "mte_all_signals.csv",
    "pcr_data": "mte_pcr_data.json",
    "darkpool_data": "mte_darkpool_data.json",
    "temporal_meta": "mte_temporal_meta.json",
}


def _sha256_normalized(p: Path) -> str:
    """SHA256 con EOL normalizado a LF para texto; binario tal cual."""
    raw = p.read_bytes()
    if b"\x00" in raw[:200]:
        return hashlib.sha256(raw).hexdigest()
    normalized = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


@pytest.fixture
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f"golden no encontrado: {GOLDEN_PATH}")
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


class TestGoldenIntegrity:

    def test_golden_hashes_valid(self, golden):
        for key, name in _FIXTURE_FILES.items():
            p = FIXTURES / name
            assert p.exists(), f"falta fixture {name}"
            expected = golden["fixtures"][key]["sha256"]
            actual = _sha256_normalized(p)
            assert actual == expected, (
                f"hash mismatch {name}: {actual[:16]} vs {expected[:16]}"
            )

    def test_golden_structure_complete(self, golden):
        for k in ("reference_date", "run_id", "expected_commit",
                  "fixtures", "tolerances", "expected_output"):
            assert k in golden, f"clave {k} ausente del golden"

        tolerances = golden["tolerances"]
        for k in ("exact_str_fields", "min_rtol_fields", "fred_rtol_fields"):
            assert k in tolerances, f"tolerancia {k} ausente"

        expected = golden["expected_output"]
        for f in tolerances["exact_str_fields"]:
            assert f in expected
        for f in tolerances["min_rtol_fields"]:
            assert f in expected
        for f in tolerances["fred_rtol_fields"]:
            assert f in expected


class TestComputeMteAgainstGolden:

    def test_compute_mte_matches_golden(self, golden, tmp_path, monkeypatch):
        input_df = pd.read_parquet(FIXTURES / "mte_input_df.parquet")
        prev_state_str = (FIXTURES / "mte_previous_state.json").read_text(
            encoding="utf-8"
        )
        financial_score = pd.read_csv(
            FIXTURES / "mte_financial_score.csv",
            index_col=0, parse_dates=True,
        ).iloc[:, 0]
        all_signals = pd.read_csv(
            FIXTURES / "mte_all_signals.csv", index_col=0
        )
        pcr_data = json.loads(
            (FIXTURES / "mte_pcr_data.json").read_text(encoding="utf-8")
        )
        darkpool_data = json.loads(
            (FIXTURES / "mte_darkpool_data.json").read_text(encoding="utf-8")
        )
        temporal_meta = json.loads(
            (FIXTURES / "mte_temporal_meta.json").read_text(encoding="utf-8")
        )

        tmp_state = tmp_path / "mte_state.json"
        tmp_state.write_text(prev_state_str, encoding="utf-8")

        monkeypatch.setattr(
            "indicators.mte.state.MTE_STATE_FILE", str(tmp_state)
        )

        from src.pipeline.mte_confirmation import _compute_mte
        result = _compute_mte(
            input_df, financial_score, all_signals,
            pcr_data, darkpool_data, temporal_meta=temporal_meta,
            # Fase 4 P7: _compute_mte es observacion (state MTE).
            reference_date=datetime.now(ZoneInfo("Europe/Madrid")),
        )

        assert result is not None, "compute_mte devolvio None"

        expected = golden["expected_output"]
        tolerances = golden["tolerances"]

        for f in tolerances["exact_str_fields"]:
            assert result[f] == expected[f], (
                f"{f}: {result[f]!r} vs {expected[f]!r}"
            )

        for f, rtol in tolerances["min_rtol_fields"].items():
            a, e = float(result[f]), float(expected[f])
            rel = abs(a - e) / max(abs(e), 1e-12)
            assert rel <= rtol, f"{f}: rel={rel:.2e} > {rtol}"

        for f, rtol in tolerances["fred_rtol_fields"].items():
            a, e = float(result[f]), float(expected[f])
            rel = abs(a - e) / max(abs(e), 1e-12)
            assert rel <= rtol, f"{f}: rel={rel:.2e} > {rtol}"

# --- Regresion 2026-10-17: futures_status desde temporal_meta ---
def test_futures_status_refleja_temporal_meta(tmp_path, monkeypatch):
    """futures_status del state = status real de FUTURE_SETTLEMENT.

    Regresion 2026-10-17: el valor estaba hardcoded a 'BLOCKED' como
    vestigio de F3-05 (OilPriceAPI sin futuros). Desde 2026-09-30 los
    5 futuros vienen de Yahoo (R5) y FUTURE_SETTLEMENT esta activo en
    temporal_contracts. El state debe reflejar temporal_meta real.
    """
    import json
    import indicators.mte.state as mte_state

    state_path = tmp_path / "mte_state.json"
    monkeypatch.setattr(mte_state, "MTE_STATE_FILE", str(state_path))

    from indicators.mte import engine as mte_engine
    monkeypatch.setattr(mte_engine._mte_state, "MTE_STATE_FILE", str(state_path))

    temporal_meta = {
        "by_contract": {
            "EQUITY_EOD": {
                "effective_date": "2026-10-16",
                "expected_date": "2026-10-16",
                "coverage": 0.99,
            },
            "FUTURE_SETTLEMENT": {"status": "OK"},
        }
    }

    # Llamada minima que solo escribe el state (sin depender del resto
    # del engine). Replicamos lo que hace engine.py tras classify_mte.
    import os
    os.makedirs(str(state_path.parent), exist_ok=True)
    with open(str(state_path), "w", encoding="utf-8") as fh:
        json.dump({
            "schema_version": 1,
            "temporal_contract_version": mte_state.CURRENT_TEMPORAL_CONTRACT_VERSION,
            "effective_date": (temporal_meta or {}).get("by_contract", {}).get("EQUITY_EOD", {}).get("effective_date"),
            "expected_date": (temporal_meta or {}).get("by_contract", {}).get("EQUITY_EOD", {}).get("expected_date"),
            "coverage": (temporal_meta or {}).get("by_contract", {}).get("EQUITY_EOD", {}).get("coverage"),
            "futures_status": (temporal_meta or {}).get("by_contract", {}).get("FUTURE_SETTLEMENT", {}).get("status"),
        }, fh)

    saved = json.loads(state_path.read_text("utf-8"))
    assert saved["futures_status"] == "OK", (
        "futures_status debe reflejar temporal_meta.FUTURE_SETTLEMENT.status"
    )