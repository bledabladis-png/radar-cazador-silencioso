# -*- coding: utf-8 -*-
"""Tests generacion de CSVs y mapping.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.annotation import (
    ANOTADORES,
    BLIND_ID_MAX,
    BLIND_ID_MIN,
    CSV_ANOTADOR_COLS,
    build_blind_muestra,
    write_all_csvs,
    write_anotador_csv,
    write_mapping_csv,
    _generar_blind_ids,
)


def _make_muestra(n=100, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    return pd.DataFrame({
        "ticker": [f"T{i:03d}" for i in range(n)],
        "t": idx,
        "muestra": rng.choice(["A", "B"], size=n),
        "sector": rng.choice(["Tech", "Health", "Energy"], size=n),
        "periodo": rng.choice(
            ["2021-2022", "2023-2024", "2025-2026"], size=n
        ),
    })


def test_generar_blind_ids_unicos_y_en_rango():
    ids = _generar_blind_ids(100, seed=1)
    assert len(np.unique(ids)) == 100
    assert ids.min() >= BLIND_ID_MIN
    assert ids.max() <= BLIND_ID_MAX


def test_generar_blind_ids_no_secuenciales():
    ids = _generar_blind_ids(100, seed=1)
    diffs = np.abs(np.diff(np.sort(ids)))
    # Esperado: todos > 1 (no secuenciales)
    assert (diffs > 1).all()


def test_generar_blind_ids_determinista():
    a = _generar_blind_ids(50, seed=7)
    b = _generar_blind_ids(50, seed=7)
    assert (a == b).all()


def test_build_blind_muestra_asigna_ids():
    m = _make_muestra(50)
    blind = build_blind_muestra(m, seed=1)
    assert len(blind.mapping) == 50
    assert blind.mapping["blind_id"].is_unique
    assert "blind_id" in blind.mapping.columns


def test_build_blind_muestra_rechaza_vacia():
    with pytest.raises(ValueError, match="vacia"):
        build_blind_muestra(pd.DataFrame(columns=["ticker", "t"]))


def test_build_blind_muestra_rechaza_duplicados():
    m = _make_muestra(10)
    m_dup = pd.concat([m, m.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicados"):
        build_blind_muestra(m_dup, seed=1)


def test_build_blind_muestra_rechaza_nulos():
    m = _make_muestra(10)
    m.loc[0, "ticker"] = None
    with pytest.raises(ValueError, match="nulos"):
        build_blind_muestra(m, seed=1)


def test_orden_por_anotador_distinto():
    m = _make_muestra(100)
    blind = build_blind_muestra(m, seed=1)
    o1 = blind.orden_por_anotador["anotador_1"]
    o2 = blind.orden_por_anotador["anotador_2"]
    o3 = blind.orden_por_anotador["anotador_3"]
    assert not np.array_equal(o1, o2)
    assert not np.array_equal(o1, o3)
    assert not np.array_equal(o2, o3)


def test_write_mapping_csv(tmp_path):
    m = _make_muestra(20)
    blind = build_blind_muestra(m, seed=1)
    p = tmp_path / "mapping.csv"
    write_mapping_csv(blind, p)
    assert p.exists()
    df = pd.read_csv(p)
    assert len(df) == 20
    for c in ("blind_id", "ticker", "t"):
        assert c in df.columns


def test_write_anotador_csv_vacio(tmp_path):
    m = _make_muestra(15)
    blind = build_blind_muestra(m, seed=1)
    p = tmp_path / "annotation_anotador_1.csv"
    write_anotador_csv(blind, "anotador_1", p)
    df = pd.read_csv(p)
    assert len(df) == 15
    assert list(df.columns) == CSV_ANOTADOR_COLS
    # Columnas vacias
    assert (df["label"].isna() | (df["label"].astype(str) == "")).all()
    assert (df["confidence"].isna() | (df["confidence"].astype(str) == "")).all()
    assert (
        df["technical_ineligible"].isna()
        | (df["technical_ineligible"].astype(str) == "")
    ).all()


def test_write_anotador_csv_orden_distinto(tmp_path):
    m = _make_muestra(50)
    blind = build_blind_muestra(m, seed=1)
    p1 = tmp_path / "a1.csv"
    p2 = tmp_path / "a2.csv"
    write_anotador_csv(blind, "anotador_1", p1)
    write_anotador_csv(blind, "anotador_2", p2)
    d1 = pd.read_csv(p1)
    d2 = pd.read_csv(p2)
    assert not d1["blind_id"].equals(d2["blind_id"])


def test_write_anotador_csv_rechaza_desconocido(tmp_path):
    m = _make_muestra(5)
    blind = build_blind_muestra(m, seed=1)
    with pytest.raises(ValueError, match="Anotador desconocido"):
        write_anotador_csv(blind, "anotador_X", tmp_path / "x.csv")


def test_write_all_csvs(tmp_path):
    m = _make_muestra(30)
    blind = build_blind_muestra(m, seed=1)
    paths = write_all_csvs(blind, tmp_path)
    assert "mapping" in paths
    for a in ANOTADORES:
        assert a in paths
        assert paths[a].exists()
    assert paths["mapping"].exists()


def test_mapping_y_anotador_no_comparten_pii(tmp_path):
    """mapping.csv tiene ticker/fecha; CSV anotador no."""
    m = _make_muestra(10)
    blind = build_blind_muestra(m, seed=1)
    paths = write_all_csvs(blind, tmp_path)
    map_df = pd.read_csv(paths["mapping"])
    an_df = pd.read_csv(paths["anotador_1"])
    assert "ticker" in map_df.columns
    assert "t" in map_df.columns
    assert "ticker" not in an_df.columns
    assert "t" not in an_df.columns
    assert "sector" not in an_df.columns
    assert "muestra" not in an_df.columns


def test_mapping_contiene_mismo_numero_de_ids(tmp_path):
    m = _make_muestra(42)
    blind = build_blind_muestra(m, seed=1)
    paths = write_all_csvs(blind, tmp_path)
    map_df = pd.read_csv(paths["mapping"])
    for a in ANOTADORES:
        an_df = pd.read_csv(paths[a])
        assert set(an_df["blind_id"]) == set(map_df["blind_id"])