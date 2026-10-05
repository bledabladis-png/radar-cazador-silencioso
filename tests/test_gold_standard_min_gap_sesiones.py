# -*- coding: utf-8 -*-
"""Tests min_gap en sesiones (P0.4, dictamen).

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.sampling_a import apply_min_gap


def test_min_gap_en_sesiones_no_dias():
    """Dos sesiones consecutivas distan 1 sesion (y pueden distar 3 dias
    si hay fin de semana). min_gap=5 debe rechazar la segunda."""
    # Crear 10 sesiones con huecos de fin de semana: 2021-01-04 (lun) a 2021-01-15 (vie)
    fechas = pd.bdate_range("2021-01-04", periods=10)
    eps = pd.DataFrame({
        "ticker": ["AAA"] * 10,
        "t": fechas,
    })
    # con min_gap=5, no puede haber dos aceptadas con menos de 5 posiciones
    out = apply_min_gap(eps, min_gap_sessions=5, seed=1)
    posiciones = [fechas.get_loc(t) for t in out["t"]]
    posiciones = sorted(posiciones)
    for a, b in zip(posiciones, posiciones[1:]):
        assert (b - a) >= 5


def test_min_gap_con_festivos():
    """120 sesiones <> 120 dias naturales cuando hay festivos."""
    # Simular un calendario con un hueco de 10 dias (festivo largo)
    fechas = pd.DatetimeIndex(
        list(pd.bdate_range("2021-01-04", periods=50)) +
        list(pd.bdate_range("2021-03-22", periods=50))
    )
    eps = pd.DataFrame({"ticker": ["AAA"] * 100, "t": fechas})
    # Con min_gap=20 sesiones
    out = apply_min_gap(eps, min_gap_sessions=20, seed=1)
    # Verificar que las sesiones aceptadas distan >= 20 en posicion ordinal
    posiciones = sorted(fechas.get_loc(t) for t in out["t"])
    for a, b in zip(posiciones, posiciones[1:]):
        assert (b - a) >= 20


def test_min_gap_por_ticker_independiente():
    fechas = pd.bdate_range("2021-01-04", periods=30)
    eps = pd.DataFrame({
        "ticker": ["AAA"] * 30 + ["BBB"] * 30,
        "t": list(fechas) + list(fechas),
    })
    out = apply_min_gap(eps, min_gap_sessions=10, seed=1)
    # Cada ticker debe cumplir por separado
    for tk in ("AAA", "BBB"):
        sub = out[out["ticker"] == tk]
        pos = sorted(fechas.get_loc(t) for t in sub["t"])
        for a, b in zip(pos, pos[1:]):
            assert (b - a) >= 10