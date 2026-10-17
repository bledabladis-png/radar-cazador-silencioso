# -*- coding: utf-8 -*-
"""REP-AUDIT-20261017-capa6-commitD: contrato del script de
migracion del CSV de regimenes.
"""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.migrate_macro_regime_csv import migrate


def _write_csv(path, df):
    df.to_csv(path, index=False)


def test_migra_solo_liquidity_regime(tmp_path):
    csv = tmp_path / "macro_regime.csv"
    _write_csv(csv, pd.DataFrame({
        "date": ["2026-10-15", "2026-10-16"],
        "macro_regime": ["MIXED", "SLOWDOWN"],
        "liquidity_regime": ["HIGH_STRESS", "NEUTRAL"],
    }))
    result = migrate(csv)
    assert result["status"] == "migrated"
    df = pd.read_csv(csv)
    assert "liquidity_regime" not in df.columns
    assert "financial_regime" in df.columns
    assert "real_liq_regime" in df.columns
    assert df["financial_regime"].tolist() == ["HIGH_STRESS", "NEUTRAL"]
    assert df["real_liq_regime"].isna().all()


def test_migra_con_ambas_columnas_identicas(tmp_path):
    csv = tmp_path / "macro_regime.csv"
    _write_csv(csv, pd.DataFrame({
        "date": ["2026-10-15"],
        "financial_regime": ["HIGH_STRESS"],
        "liquidity_regime": ["HIGH_STRESS"],
    }))
    result = migrate(csv)
    assert result["status"] == "migrated"
    df = pd.read_csv(csv)
    assert "liquidity_regime" not in df.columns
    assert df.iloc[0]["financial_regime"] == "HIGH_STRESS"


def test_aborta_si_columnas_difieren(tmp_path):
    csv = tmp_path / "macro_regime.csv"
    _write_csv(csv, pd.DataFrame({
        "date": ["2026-10-15"],
        "financial_regime": ["HIGH_STRESS"],
        "liquidity_regime": ["NEUTRAL"],
    }))
    with pytest.raises(ValueError, match="difieren"):
        migrate(csv)


def test_idempotente(tmp_path):
    csv = tmp_path / "macro_regime.csv"
    _write_csv(csv, pd.DataFrame({
        "date": ["2026-10-15"],
        "liquidity_regime": ["ESTRECHA"],
    }))
    r1 = migrate(csv)
    assert r1["status"] == "migrated"
    r2 = migrate(csv)
    assert r2["status"] == "already_migrated"


def test_not_found(tmp_path):
    csv = tmp_path / "no_existe.csv"
    r = migrate(csv)
    assert r["status"] == "not_found"


def test_dry_run_no_escribe(tmp_path):
    csv = tmp_path / "macro_regime.csv"
    _write_csv(csv, pd.DataFrame({
        "date": ["2026-10-15"],
        "liquidity_regime": ["ESTRECHA"],
    }))
    r = migrate(csv, dry_run=True)
    assert r["status"] == "dry_run"
    df = pd.read_csv(csv)
    assert "liquidity_regime" in df.columns
    assert "financial_regime" not in df.columns
