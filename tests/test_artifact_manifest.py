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


def test_fu_002_evo_valid_with_missing(tmp_path):
    """FU-002-evo (2026-09-15): huecos legitimos -> VALID_WITH_MISSING.

    Con close_nan_last > 0 (por B1) y pct_dup_last bajo -> no es corrupcion.
    """
    df = _make_df(last_date='2026-09-15', n_tickers=5, add_nan_last=True)
    parquet_path = tmp_path / 'missing.parquet'
    ref_date = datetime(2026, 9, 15, 23, 30)

    manifest = write_artifact_with_manifest(
        df, str(parquet_path),
        source='test', reference_date=ref_date, run_id='test_vwm',
    )

    assert manifest['quality']['status'] == 'VALID_WITH_MISSING'
    assert manifest['quality']['close_nan_last'] > 0
    assert manifest['quality']['pct_dup_last'] == 0.0
    assert manifest['quality']['last_date_is_expected_session'] is True


def test_fu_002_evo2_valid_with_missing_pre_publish_no_contract(tmp_path):
    """FU-002-evo2, sin contrato temporal activado: close_nan > 0 con
    last_date != expected_session -> VALID_WITH_MISSING.

    Con temporal_contract=None la validacion temporal no aplica: este
    test preserva el comportamiento historico de FU-002-evo2. Cuando
    FU-021-5 active un contrato explicito, un last_date > expected_session
    pasara a INVALID.
    """
    df = _make_df(last_date='2026-09-15', n_tickers=5, add_nan_last=True)
    parquet_path = tmp_path / 'prepub.parquet'
    ref_date = datetime(2026, 9, 15, 10, 35)

    manifest = write_artifact_with_manifest(
        df, str(parquet_path),
        source='test', reference_date=ref_date, run_id='test_evo2',
    )

    assert manifest['quality']['last_date'] == '2026-09-15'
    assert manifest['quality']['expected_session'] == '2026-09-14'
    assert manifest['quality']['last_date_is_expected_session'] is False
    assert manifest['quality']['close_nan_last'] > 0
    assert manifest['quality']['pct_dup_last'] == 0.0
    assert manifest['quality']['status'] == 'VALID_WITH_MISSING'
    assert manifest['quality']['last_row_is_partial'] is True
    assert manifest['quality']['n_tickers_with_close_last'] < 5
    assert manifest['quality']['n_tickers_missing_close_last'] > 0
    assert manifest['quality']['coverage_pct_last'] < 1.0


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

# --- FU-002-bis: temporal_contract ---


def test_temporal_contract_none_no_temporal_validation(tmp_path):
    """Sin contrato: comportamiento FU-002-evo2 (sin validacion temporal).

    last_date > expected_session con close_nan=0 y contract=None -> VALID.
    """
    df = _make_df(last_date='2026-09-15', dup_ratio=0.0)
    parquet_path = tmp_path / 'no_contract.parquet'
    ref_date = datetime(2026, 9, 15, 10, 35)

    manifest = write_artifact_with_manifest(
        df, str(parquet_path),
        source='test', reference_date=ref_date, run_id='test_nc',
        temporal_contract=None,
    )

    assert manifest['quality']['last_date'] == '2026-09-15'
    assert manifest['quality']['expected_session'] == '2026-09-14'
    assert manifest['quality']['last_date_is_expected_session'] is False
    assert manifest['temporal']['contract'] is None
    # Sin contrato: no aplica regla temporal -> VALID (close_nan=0)
    assert manifest['quality']['status'] == 'VALID'


def test_temporal_contract_equity_eod_rejects_future_date(tmp_path):
    """Con contrato EQUITY_EOD: last_date > expected_session -> INVALID.

    close_nan=0, pct_dup=0. Sin embargo, violacion temporal -> INVALID.
    """
    df = _make_df(last_date='2026-09-15', dup_ratio=0.0)
    parquet_path = tmp_path / 'contract_future.parquet'
    ref_date = datetime(2026, 9, 15, 10, 35)

    manifest = write_artifact_with_manifest(
        df, str(parquet_path),
        source='test', reference_date=ref_date, run_id='test_contract_f',
        temporal_contract='EQUITY_EOD',
    )

    assert manifest['temporal']['contract'] == 'EQUITY_EOD'
    assert manifest['quality']['last_date_is_expected_session'] is False
    assert manifest['quality']['status'] == 'INVALID'


def test_temporal_contract_equity_eod_rejects_future_date_with_nan(tmp_path):
    """Con contrato: last_date > expected_session + close_nan>0 -> INVALID.

    Evita que la logica tipo Opcion B (future+nan -> VALID_WITH_MISSING)
    reaparezca accidentalmente. La temporalidad y la completitud son
    dimensiones independientes.
    """
    df = _make_df(last_date='2026-09-15', n_tickers=5, add_nan_last=True)
    parquet_path = tmp_path / 'contract_future_nan.parquet'
    ref_date = datetime(2026, 9, 15, 10, 35)

    manifest = write_artifact_with_manifest(
        df, str(parquet_path),
        source='test', reference_date=ref_date, run_id='test_contract_fn',
        temporal_contract='EQUITY_EOD',
    )

    assert manifest['quality']['close_nan_last'] > 0
    assert manifest['quality']['last_date_is_expected_session'] is False
    assert manifest['quality']['status'] == 'INVALID'


def test_temporal_contract_declared_recorded_in_manifest(tmp_path):
    """El manifest registra temporal.contract == 'EQUITY_EOD'."""
    df = _make_df(last_date='2026-09-15', dup_ratio=0.0)
    parquet_path = tmp_path / 'contract_ok.parquet'
    ref_date = datetime(2026, 9, 15, 23, 30)

    manifest = write_artifact_with_manifest(
        df, str(parquet_path),
        source='test', reference_date=ref_date, run_id='test_contract_ok',
        temporal_contract='EQUITY_EOD',
    )

    assert manifest['temporal']['contract'] == 'EQUITY_EOD'
    assert manifest['quality']['last_date_is_expected_session'] is True
    assert manifest['quality']['status'] == 'VALID'

    manifest_file = Path(str(parquet_path) + '.manifest.json')
    on_disk = json.loads(manifest_file.read_text(encoding='utf-8'))
    assert on_disk['temporal']['contract'] == 'EQUITY_EOD'


def test_temporal_contract_none_recorded_in_manifest(tmp_path):
    """El manifest registra temporal.contract == None (siempre presente)."""
    df = _make_df(last_date='2026-09-15', dup_ratio=0.0)
    parquet_path = tmp_path / 'no_contract_manifest.parquet'
    ref_date = datetime(2026, 9, 15, 23, 30)

    manifest = write_artifact_with_manifest(
        df, str(parquet_path),
        source='test', reference_date=ref_date, run_id='test_nc_m',
    )

    assert 'temporal' in manifest
    assert manifest['temporal']['contract'] is None

    manifest_file = Path(str(parquet_path) + '.manifest.json')
    on_disk = json.loads(manifest_file.read_text(encoding='utf-8'))
    assert 'temporal' in on_disk
    assert on_disk['temporal']['contract'] is None

# --- FU-021-3C-bis: bugfix close_cols con DataFrame de 1 fila ---


def test_manifest_single_row_returns_dict(tmp_path):
    """FU-021-3C-bis: df con 1 fila -> manifest se escribe y se retorna.

    Bug previo: close_cols se definia dentro de un bloque condicional
    (len(df) >= 2) pero se usaba fuera (en el print de observabilidad).
    Con df de 1 fila, UnboundLocalError silencioso: el manifest se
    escribia en disco correctamente, pero la funcion retornaba {}.
    Afectaba a writers con df corto (commodities_*.parquet).

    Verifica: returns dict no vacio, status VALID, sha256 presente,
    fichero en disco correcto.
    """
    df = _make_df_single_row(last_date='2026-09-15', n_tickers=2)
    parquet_path = tmp_path / 'single.parquet'
    ref_date = datetime(2026, 9, 15, 23, 30)

    manifest = write_artifact_with_manifest(
        df, str(parquet_path),
        source='test_single',
        reference_date=ref_date,
        run_id='test_single_row',
    )

    # Este assert es el bugfix: con el bug, manifest == {}.
    assert manifest != {}, "manifest retorno {} con df de 1 fila (bug close_cols)"
    assert manifest['schema_version'] == 1
    assert manifest['quality']['status'] == 'VALID'
    assert manifest['content']['rows'] == 1
    assert manifest['content']['n_tickers'] == 2
    assert manifest['quality']['last_date'] == '2026-09-15'
    assert manifest['quality']['close_nan_last'] == 0
    assert manifest['artifact']['sha256']

    manifest_file = Path(str(parquet_path) + '.manifest.json')
    assert manifest_file.exists(), "manifest no escrito en disco"


def _make_df_single_row(last_date='2026-09-15', n_tickers=2):
    """DataFrame MultiIndex con 1 sola fila. Reproduce el caso commodities."""
    dates = pd.to_datetime([last_date])
    tickers = [f'T{i}' for i in range(n_tickers)]
    data = {}
    for i, t in enumerate(tickers):
        base = 100 + i
        data[('Close', t)] = [base]
        data[('Open', t)] = [base - 1]
        data[('High', t)] = [base + 2]
        data[('Low', t)] = [base - 2]
        data[('Volume', t)] = [1_000_000]
    return pd.DataFrame(data, index=dates)
