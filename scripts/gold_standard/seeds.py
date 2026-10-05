# -*- coding: utf-8 -*-
"""Generacion determinista de seeds por replica.

Dictamen P1.3: workers=1 y workers=12 deben producir exactamente
el mismo resultado. La seed de cada replica se deriva de un master
seed mediante SeedSequence, sin depender del orden de ejecucion.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import numpy as np


def replica_seeds(master_seed: int, B: int) -> list[int]:
    """Genera B seeds deterministas a partir de master_seed.

    Uso: cada replica b usa replica_seeds(master, B)[b].
    Independiente del numero de workers.
    """
    if B <= 0:
        return []
    ss = np.random.SeedSequence(master_seed)
    hijos = ss.spawn(B)
    return [int(h.generate_state(1)[0]) for h in hijos]


def rng_for_replica(master_seed: int, b: int) -> np.random.Generator:
    """RNG determinista para la replica b."""
    ss = np.random.SeedSequence(master_seed)
    hijos = ss.spawn(b + 1)
    return np.random.default_rng(hijos[-1])