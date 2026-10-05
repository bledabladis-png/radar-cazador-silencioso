# -*- coding: utf-8 -*-
"""Tests separacion ADMIN/ANNOTATOR (P1.4, dictamen).

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
    build_blind_muestra,
    write_paquetes_anotadores,
)


def _make_muestra(n=30, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    return pd.DataFrame({
        "ticker": [f"T{i:03d}" for i in range(n)],
        "t": idx,
        "muestra": rng.choice(["A", "B"], size=n),
        "sector": rng.choice(["Tech", "Health"], size=n),
        "periodo": rng.choice(["2021-2022", "2023-2024"], size=n),
    })


def test_estructura_admin_annotator(tmp_path):
    m = _make_muestra(20)
    blind = build_blind_muestra(m, seed=1)
    write_paquetes_anotadores(blind, tmp_path)

    # ADMIN existe
    assert (tmp_path / "ADMIN").exists()
    assert (tmp_path / "ADMIN" / "mapping.csv").exists()
    assert (tmp_path / "ADMIN" / "manifest_admin.json").exists()

    # ANNOTATOR_1/2/3 existen
    for i, nombre in enumerate(("anotador_1", "anotador_2", "anotador_3"), 1):
        an_dir = tmp_path / f"ANNOTATOR_{i}"
        assert an_dir.exists()
        assert (an_dir / f"annotation_{nombre}.csv").exists()


def test_annotator_no_contiene_mapping(tmp_path):
    m = _make_muestra(10)
    blind = build_blind_muestra(m, seed=1)
    write_paquetes_anotadores(blind, tmp_path)

    for i in (1, 2, 3):
        an_dir = tmp_path / f"ANNOTATOR_{i}"
        assert not (an_dir / "mapping.csv").exists()
        assert not (an_dir / "manifest_admin.json").exists()


def test_annotator_csv_sin_pii(tmp_path):
    m = _make_muestra(10)
    blind = build_blind_muestra(m, seed=1)
    write_paquetes_anotadores(blind, tmp_path)

    for i, nombre in enumerate(("anotador_1", "anotador_2", "anotador_3"), 1):
        csv = tmp_path / f"ANNOTATOR_{i}" / f"annotation_{nombre}.csv"
        df = pd.read_csv(csv)
        assert "ticker" not in df.columns
        assert "t" not in df.columns
        assert "sector" not in df.columns
        assert "blind_id" in df.columns


def test_admin_y_annotator_mismo_n(tmp_path):
    m = _make_muestra(25)
    blind = build_blind_muestra(m, seed=1)
    write_paquetes_anotadores(blind, tmp_path)

    admin_df = pd.read_csv(tmp_path / "ADMIN" / "mapping.csv")
    for i, nombre in enumerate(("anotador_1", "anotador_2", "anotador_3"), 1):
        an_df = pd.read_csv(
            tmp_path / f"ANNOTATOR_{i}" / f"annotation_{nombre}.csv"
        )
        assert len(an_df) == len(admin_df)
        assert set(an_df["blind_id"]) == set(admin_df["blind_id"])