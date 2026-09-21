"""Tests S - contrato de entrada B1 <-> B2-PIT (seccion 8)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.institutional_accumulation.identity import catalog_key as ck


def _snap(columns):
    return pd.DataFrame(
        [{c: "x" for c in columns}],
        columns=list(columns),
    )

# --- S-a: snapshot con ambas columnas -> OK ---

def test_Sa_ambas_columnas():
    df = _snap(["radar_ticker", "share_class_figi", "name"])
    assert ck.check_schema(df) is True


# --- S-b: snapshot sin radar_ticker -> B1_SCHEMA_ERROR ---

def test_Sb_sin_radar_ticker():
    df = _snap(["share_class_figi", "name"])
    with pytest.raises(ValueError, match=ck.ERR_B1_SCHEMA_ERROR):
        ck.check_schema(df)


# --- S-c: snapshot sin share_class_figi -> B1_SCHEMA_ERROR ---

def test_Sc_sin_share_class_figi():
    df = _snap(["radar_ticker", "name"])
    with pytest.raises(ValueError, match=ck.ERR_B1_SCHEMA_ERROR):
        ck.check_schema(df)


# --- S-d: snapshot sin ambas -> B1_SCHEMA_ERROR ---

def test_Sd_sin_ambas():
    df = _snap(["name", "source_date"])
    with pytest.raises(ValueError, match=ck.ERR_B1_SCHEMA_ERROR):
        ck.check_schema(df)


# --- S-e: snapshot con columnas extra -> OK (ignoradas) ---

def test_Se_columnas_extra_ok():
    df = _snap(["radar_ticker", "share_class_figi", "name", "extra1", "extra2"])
    assert ck.check_schema(df) is True


# --- Sanity: B1_REQUIRED_COLUMNS contenido ---

def test_required_columns_contenido():
    assert ck.B1_REQUIRED_COLUMNS == frozenset({"radar_ticker", "share_class_figi"})


# --- Snapshot real cumple schema ---

def test_snapshot_real_cumple_schema():
    df = pd.read_csv(
        "data/mappings/catalog_snapshots/snapshot_20260921_01.csv",
        dtype=str, keep_default_na=False,
    )
    assert ck.check_schema(df) is True