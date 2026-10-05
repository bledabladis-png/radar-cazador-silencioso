# -*- coding: utf-8 -*-
"""Test: run_capa1._finalize produce estructura ADMIN/ANNOTATOR.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.annotation import (
    build_blind_muestra, write_paquetes_anotadores,
)


def _make_muestra(n=10, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ticker": [f"T{i:03d}" for i in range(n)],
        "t": pd.date_range("2021-01-04", periods=n, freq="B"),
        "muestra": rng.choice(["A", "B"], size=n),
        "sector": "Tech",
        "periodo": "2021-2022",
    })


def test_estructura_con_cases_master(tmp_path):
    m = _make_muestra(8)
    blind = build_blind_muestra(m, seed=1)

    # Simular cases_master con PNGs vacios
    cases_master = tmp_path / "_cases_master"
    cases_master.mkdir()
    for bid in blind.mapping["blind_id"]:
        (cases_master / f"case_{bid}.png").write_bytes(b"fake")

    out = tmp_path / "out"
    write_paquetes_anotadores(
        blind, out, cases_dir_source=cases_master,
    )

    # Verificar estructura
    assert (out / "ADMIN" / "mapping.csv").exists()
    assert (out / "ADMIN" / "manifest_admin.json").exists()
    for i, nombre in enumerate(("anotador_1", "anotador_2", "anotador_3"), 1):
        an_dir = out / f"ANNOTATOR_{i}"
        assert (an_dir / f"annotation_{nombre}.csv").exists()
        cases = an_dir / "cases"
        assert cases.exists()
        assert len(list(cases.glob("case_*.png"))) == 8

    # Ningun ANNOTATOR tiene mapping
    for i in (1, 2, 3):
        assert not (out / f"ANNOTATOR_{i}" / "mapping.csv").exists()


def test_sin_cases_master_solo_csv(tmp_path):
    m = _make_muestra(5)
    blind = build_blind_muestra(m, seed=1)
    out = tmp_path / "out"
    write_paquetes_anotadores(blind, out)
    for i in (1, 2, 3):
        assert not (out / f"ANNOTATOR_{i}" / "cases").exists()