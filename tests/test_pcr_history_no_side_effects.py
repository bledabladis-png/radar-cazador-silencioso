# -*- coding: utf-8 -*-
"""Test: compute_pcr_signals() no debe escribir pcr_history.csv sin flag.

Item: REP-AUDIT-20261022-PCR-HISTORY-SIDE-EFFECT
Causa: compute_pcr_signals() hacia os.replace() a pcr_history.csv
como efecto secundario de su calculo. Bypaseaba output_writer.persist,
no emitia WriteReceipt, no se auditaba. Un diagnostico (llamar la
funcion) modificaba el working tree.
"""

import hashlib
from pathlib import Path

import pytest

PCR_PATH = Path("outputs/history/pcr_history.csv")


def _hash(path):
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compute_pcr_signals_sin_flag_no_escribe(tmp_path, monkeypatch):
    """Llamar compute_pcr_signals() sin flag NO modifica el CSV."""
    if not PCR_PATH.exists():
        pytest.skip("pcr_history.csv no existe en este entorno")

    # Leer el contenido ANTES del chdir (despues, PCR_PATH se resuelve
    # contra tmp_path y no existe).
    original_bytes = PCR_PATH.read_bytes()

    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True, exist_ok=True)
    target = tmp_path / "outputs" / "history" / "pcr_history.csv"
    target.write_bytes(original_bytes)
    before = _hash(target)

    from indicators.options import compute_pcr_signals
    try:
        compute_pcr_signals()
    except Exception:
        # Si el provider CBOE no esta disponible, la funcion puede fallar,
        # pero NO debe haber escrito.
        pass

    after = _hash(target)
    assert before == after, "compute_pcr_signals() modifico el CSV sin flag"


def test_firma_acepta_persist_history():
    """La firma expone el flag persist_history (default False)."""
    import inspect
    from indicators.options import compute_pcr_signals
    sig = inspect.signature(compute_pcr_signals)
    assert "persist_history" in sig.parameters
    assert sig.parameters["persist_history"].default is False


def test_registry_tiene_pcr_history():
    """pcr_history esta en output_registry con merge_keys=['date']."""
    from src.pipeline.output_registry import get_spec
    spec = get_spec("pcr_history")
    assert spec.kind == "temporal"
    assert spec.date_column == "date"
    assert "date" in spec.merge_keys
    assert spec.date_source == "upstream"