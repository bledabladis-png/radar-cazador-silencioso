"""Paquete MTE (DT2 Fase 2).

Re-exporta el modulo original (ahora en mte_legacy.py) preservando
exactamente la API publica. El alias sys.modules garantiza que
'import indicators.mte' devuelve el modulo legacy, manteniendo
identidad para monkeypatch (MTE_STATE_FILE, CURRENT_TEMPORAL_CONTRACT_VERSION).

Fases 3-6 (extraccion progresiva) sustituiran mte_legacy.py por
submodulos especificos: state.py, scoring.py, decision.py, engine.py.
"""
from __future__ import annotations

import sys

from . import mte_legacy

# Alias: 'indicators.mte' ES 'indicators.mte.mte_legacy'.
sys.modules[__name__] = mte_legacy