# tests/test_artifact_manifest.py
"""Tests FU-002 Commit 1 (2026-09-15): manifest de artefacto.

Cubre:
- FU-002-1: parquet + manifest valido -> status VALID
- FU-002-4: duplicacion > threshold en sesion esperada -> status INVALID
- FU-002-5: fallo escribiendo manifest -> parquet queda, sin excepcion
- Determinismo: reference_date fija -> expected_session determinista
"""
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils import write_artifact_with_manifest


def _make_df(last_date='2026-09-15', n_tickers=5, dup_ratio=0.0, add_nan_last=False):
    """DataFrame sintetico MultiIndex con columnas OHLCV x tickers."""
    dates = pd.date_range(end=last_date, periods=100, freq='D')
    tickers = [f'T{i}' for i in range(n_tickers)]
    data = {}
    for i, t in enumerate(tickers):
        base = 100 + i
        close_vals = np.linspace(base, base + 10, 100).tolist()
        if i < int(n_tickers * dup_ratio):
            close_vals[-1] = close_vals[-2]
        if add_nan_last and i == 0:
            close_vals[-1] = np.nan
        data[('Close', t)] = close_vals
        data[('Open', t)] = [v - 1 for v in close_vals]
        data[('High', t)] = [v + 2 for v in close_vals]
        data[('Low', t)] = [v - 2 for v in close_vals]
        data[('Volume', t)] = [1_000_000] * 100
    return pd.DataFrame(data, index=dates)


def test_fu_002_1_manifest_valid(tmp_path):
    """FU-002-1: parquet + manifest valido -> status VALID."""
    df = _make_df(last_date='2026-09-15', dup_ratio=0.0)
    parquet_path = tmp_path / 'valid.parquet'
    ref_date = datetime(2026, 9, 15, 23, 30)

    manifest = write_artifact_with_manifest(
        df, str(parquet_path),
        source='test', reference_date=ref_date, run_id='test001',
    )

    assert manifest, "manifest no escrito"
    assert manifest['schema_version'] == 1
    assert manifest['quality']['status'] == 'VALID'
    assert manifest['quality']['last_date'] == '2026-09-15'
    assert manifest['quality']['expected_session'] == '2026-09-15'
    assert manifest['quality']['last_date_is_expected_session'] is True
    assert manifest['content']['n_tickers'] == 5

    manifest_file = Path(str(parquet_path) + '.manifest.json')
    assert manifest_file.exists()
    on_disk = json.loads(manifest_file.read_text(encoding='utf-8'))
    assert on_disk['artifact']['sha256'] == manifest['artifact']['sha256']


def test_fu_002_4_manifest_invalid_dup(tmp_path):
    """FU-002-4: duplicacion > threshold en sesion esperada -> INVALID."""
    df = _make_df(last_date='2026-09-15', dup_ratio=0.8)
    parquet_path = tmp_path / 'dup.parquet'
    ref_date = datetime(2026, 9, 15, 23, 30)

    manifest = write_artifact_with_manifest(
        df, str(parquet_path),
        source='test', reference_date=ref_date, run_id='test004',
    )

    assert manifest['quality']['pct_dup_last'] > 0.5
    assert manifest['quality']['last_date_is_expected_session'] is True
    assert manifest['quality']['status'] == 'INVALID'


def test_fu_002_5_manifest_write_fail(tmp_path, monkeypatch):
    """FU-002-5: fallo escribiendo manifest -> parquet queda, sin excepcion."""
    df = _make_df(last_date='2026-09-15', dup_ratio=0.0)
    parquet_path = tmp_path / 'fail.parquet'
    ref_date = datetime(2026, 9, 15, 23, 30)

    real_replace = os.replace
    def fake_replace(src, dst):
        if str(dst).endswith('.manifest.json'):
            raise OSError('simulated manifest write failure')
        return real_replace(src, dst)
    monkeypatch.setattr(os, 'replace', fake_replace)

    manifest = write_artifact_with_manifest(
        df, str(parquet_path),
        source='test', reference_date=ref_date, run_id='test005',
    )

    assert manifest == {}
    assert parquet_path.exists(), "parquet debe permanecer tras fallo del manifest"
    manifest_file = Path(str(parquet_path) + '.manifest.json')
    assert not manifest_file.exists()


def test_manifest_expected_session_deterministic(tmp_path):
    """reference_date fija -> expected_session determinista entre runs."""
    df = _make_df(last_date='2026-09-15', dup_ratio=0.0)
    ref_date = datetime(2026, 9, 15, 23, 30)

    m1 = write_artifact_with_manifest(df, str(tmp_path / 'a.parquet'),
                                       source='test', reference_date=ref_date, run_id='r1')
    m2 = write_artifact_with_manifest(df, str(tmp_path / 'b.parquet'),
                                       source='test', reference_date=ref_date, run_id='r2')

    assert m1['quality']['expected_session'] == m2['quality']['expected_session']
    assert m1['quality']['expected_session'] == '2026-09-15'
