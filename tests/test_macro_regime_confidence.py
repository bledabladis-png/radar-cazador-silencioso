# -*- coding: utf-8 -*-
"""D13 (2026-09-30): confianza real en compute_macro_regime.

Antes: conf = 0.5 hardcode. El aviso de header.py (macro_conf < 0.30)
nunca se disparaba. Ahora: confidence_from_range (misma politica que
financial_conditions, C19).

Los tests no construyen un df_market completo (fragil: compute_macro_regime
consume curve, credit, breadth y muchos otros). Verifican el contrato
en la fuente: 1 test de regresion estatica + tests del calculo de conf
sobre all_signals sintetico.
"""
from pathlib import Path

import pandas as pd

from src.utils import confidence_from_range


REPO_ROOT = Path(__file__).resolve().parents[1]
MACRO_REGIME_SRC = REPO_ROOT / "regimes" / "macro_regime.py"


def test_macro_regime_no_hardcodea_conf():
    """Regresion estatica: no volver a poner conf = 0.5 como codigo.

    Excluye lineas de comentario (que si mencionan el hardcode original
    a modo de documentacion).
    """
    lines = MACRO_REGIME_SRC.read_text(encoding="utf-8").splitlines()
    code_lines = [ln for ln in lines if not ln.lstrip().startswith("#")]
    code_text = "\n".join(code_lines)
    assert "conf = 0.5" not in code_text, "regresion: conf = 0.5 en codigo"
    assert "confidence_from_range" in code_text, "falta el calculo real"


def test_conf_calculo_signals_acordes():
    """Senales acordes -> disagreement cero -> conf 1.0."""
    idx = pd.date_range("2026-01-01", periods=10)
    signals = pd.DataFrame(
        {"a": [0.5] * 10, "b": [0.5] * 10, "c": [0.5] * 10, "d": [0.5] * 10},
        index=idx,
    )
    last = signals.iloc[-1].dropna()
    conf = float(confidence_from_range(last, divisor=2.0))
    assert conf == 1.0


def test_conf_calculo_signals_opuestas():
    """Senales opuestas extremas -> disagreement 2.0 -> conf 0.0."""
    idx = pd.date_range("2026-01-01", periods=10)
    signals = pd.DataFrame(
        {"a": [-1.0] * 10, "b": [1.0] * 10, "c": [-1.0] * 10, "d": [1.0] * 10},
        index=idx,
    )
    last = signals.iloc[-1].dropna()
    conf = float(confidence_from_range(last, divisor=2.0))
    assert conf == 0.0


def test_conf_calculo_una_sola_senal():
    """Con <2 senales validas -> 0.5 (evidencia insuficiente)."""
    idx = pd.date_range("2026-01-01", periods=10)
    signals = pd.DataFrame(
        {"a": [0.5] * 10, "b": [float("nan")] * 10}, index=idx
    )
    last = signals.iloc[-1].dropna()
    conf = float(confidence_from_range(last, divisor=2.0))
    assert conf == 0.5


def test_conf_calculo_rango_valido():
    """Contrato: conf en [0, 1]."""
    idx = pd.date_range("2026-01-01", periods=10)
    signals = pd.DataFrame(
        {"a": [-0.3] * 10, "b": [0.7] * 10, "c": [0.1] * 10}, index=idx
    )
    last = signals.iloc[-1].dropna()
    conf = float(confidence_from_range(last, divisor=2.0))
    assert 0.0 <= conf <= 1.0
