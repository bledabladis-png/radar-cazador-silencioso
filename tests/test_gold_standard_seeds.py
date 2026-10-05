# -*- coding: utf-8 -*-
"""Tests seeds deterministas (P1.3, dictamen).

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.seeds import replica_seeds, rng_for_replica


def test_replica_seeds_determinista():
    a = replica_seeds(20261006, 10)
    b = replica_seeds(20261006, 10)
    assert a == b


def test_replica_seeds_distintas():
    s = replica_seeds(20261006, 100)
    assert len(set(s)) == 100


def test_replica_seeds_B_cero():
    assert replica_seeds(20261006, 0) == []


def test_rng_for_replica_determinista():
    r1 = rng_for_replica(42, 5)
    r2 = rng_for_replica(42, 5)
    a1 = r1.integers(0, 1000, size=10).tolist()
    a2 = r2.integers(0, 1000, size=10).tolist()
    assert a1 == a2


def test_rng_distintas_replicas_distintos_valores():
    r1 = rng_for_replica(42, 1)
    r2 = rng_for_replica(42, 2)
    a1 = r1.integers(0, 10**9, size=10).tolist()
    a2 = r2.integers(0, 10**9, size=10).tolist()
    assert a1 != a2