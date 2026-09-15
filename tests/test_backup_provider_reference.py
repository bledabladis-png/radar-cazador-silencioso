# tests/test_backup_provider_reference.py
"""Tests FU-002 Commit 2: consumer rechaza cache no verificado.

Cubre:
- FU-002-2: sha256 modificado -> INVALID
- FU-002-3: manifest ausente -> UNAVAILABLE
- FU-002-6: discrepancia >5% contra VALID -> False
- FU-002-7: cache INVALID no legitima (resultado None)
- FU-002-8: cache VALID + provider OK -> True
- FU-002-9: manifest corrupto -> UNAVAILABLE, sin excepcion
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils import write_artifact_with_manifest


def _make_df(last_date='2026-09-15', n_tickers=5, dup_ratio=0.0, add_nan_last=False):
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


@pytest.fixture
def patched_paths(tmp_path, monkeypatch):
    market_p = tmp_path / 'market_data.parquet'
    stocks_p = tmp_path / 'stock_prices.parquet'
    monkeypatch.setattr(
        'data.providers.backup_providers.CACHE_MARKET_PATH',
        str(market_p),
    )
    monkeypatch.setattr(
        'data.providers.backup_providers.CACHE_STOCKS_PATH',
        str(stocks_p),
    )
    return market_p, stocks_p


def _setup_valid_reference(market_p, stocks_p, ref_date):
    write_artifact_with_manifest(_make_df(dup_ratio=0.0), str(market_p),
                                  source='test', reference_date=ref_date, run_id='r_market')
    write_artifact_with_manifest(_make_df(dup_ratio=0.0), str(stocks_p),
                                  source='test', reference_date=ref_date, run_id='r_stocks')


def test_fu_002_evo_valid_with_missing_accepted(patched_paths):
    """FU-002-evo (2026-09-15): cache VALID_WITH_MISSING se acepta como VALID."""
    from data.providers.backup_providers import BackupProvider
    market_p, stocks_p = patched_paths
    ref_date = datetime(2026, 9, 15, 23, 30)
    write_artifact_with_manifest(_make_df(add_nan_last=True), str(market_p),
                                  source='test', reference_date=ref_date, run_id='r_market')
    write_artifact_with_manifest(_make_df(dup_ratio=0.0), str(stocks_p),
                                  source='test', reference_date=ref_date, run_id='r_stocks')

    bp = BackupProvider()
    assert bp.reference_cache_status == 'VALID', (
        f"cache con huecos legitimos deberia ser VALID, no {bp.reference_cache_status}"
    )


def test_fu_002_3_manifest_missing_unavailable(patched_paths):
    """FU-002-3: parquet sin manifest -> UNAVAILABLE."""
    from data.providers.backup_providers import BackupProvider
    market_p, stocks_p = patched_paths
    _make_df(dup_ratio=0.0).to_parquet(market_p)
    bp = BackupProvider()
    assert bp.reference_cache_status == 'UNAVAILABLE'


def test_fu_002_2_sha_mismatch_invalid(patched_paths):
    """FU-002-2: sha256 modificado -> INVALID."""
    from data.providers.backup_providers import BackupProvider
    market_p, stocks_p = patched_paths
    ref_date = datetime(2026, 9, 15, 23, 30)
    _setup_valid_reference(market_p, stocks_p, ref_date)

    # Modificar el parquet tras escribir el manifest
    df = pd.read_parquet(market_p)
    df = df * 1.001
    df.to_parquet(market_p)

    bp = BackupProvider()
    assert bp.reference_cache_status == 'INVALID'


def test_fu_002_9_manifest_corrupt_unavailable(patched_paths):
    """FU-002-9: manifest malformed -> UNAVAILABLE, sin excepcion."""
    from data.providers.backup_providers import BackupProvider
    market_p, stocks_p = patched_paths
    _make_df(dup_ratio=0.0).to_parquet(market_p)
    manifest_p = Path(str(market_p) + '.manifest.json')
    manifest_p.write_text('{ this is not valid json', encoding='utf-8')

    bp = BackupProvider()
    assert bp.reference_cache_status == 'UNAVAILABLE'


def test_fu_002_7_cache_invalid_not_legitimizing(patched_paths):
    """FU-002-7: cache INVALID no legitima; provider recibe None."""
    from data.providers.backup_providers import BackupProvider
    market_p, stocks_p = patched_paths
    ref_date = datetime(2026, 9, 15, 23, 30)
    write_artifact_with_manifest(_make_df(dup_ratio=0.8), str(market_p),
                                  source='test', reference_date=ref_date, run_id='r_market')
    write_artifact_with_manifest(_make_df(dup_ratio=0.0), str(stocks_p),
                                  source='test', reference_date=ref_date, run_id='r_stocks')

    bp = BackupProvider()
    assert bp.reference_cache_status == 'INVALID'

    df_new = _make_df(dup_ratio=0.0)
    result = bp._validate_with_cache('T0', df_new)
    assert result is None


def test_fu_002_8_clean_cache_valid(patched_paths):
    """FU-002-8: cache limpio + provider OK -> True."""
    from data.providers.backup_providers import BackupProvider
    market_p, stocks_p = patched_paths
    ref_date = datetime(2026, 9, 15, 23, 30)
    _setup_valid_reference(market_p, stocks_p, ref_date)

    bp = BackupProvider()
    assert bp.reference_cache_status == 'VALID'

    df_new = _make_df(dup_ratio=0.0)
    result = bp._validate_with_cache('T0', df_new)
    assert result is True


def test_fu_002_6_discrepancy_rejected(patched_paths):
    """FU-002-6: discrepancia >5% contra referencia VALID -> False."""
    from data.providers.backup_providers import BackupProvider
    market_p, stocks_p = patched_paths
    ref_date = datetime(2026, 9, 15, 23, 30)
    _setup_valid_reference(market_p, stocks_p, ref_date)

    bp = BackupProvider()
    assert bp.reference_cache_status == 'VALID'

    # Modificar Close del ticker T0 en +10%
    df_new = _make_df(dup_ratio=0.0)
    df_new[('Close', 'T0')] = df_new[('Close', 'T0')] * 1.10

    result = bp._validate_with_cache('T0', df_new)
    assert result is False


# --- FU-002-bis: writer <-> reader con temporal_contract ---


def test_contract_none_reader_accepts(patched_paths):
    """Sin contrato: writer produce VALID, reader acepta.

    Compatibilidad transicional: contract=None mantiene la semantica
    historica de FU-002-evo2.
    """
    from data.providers.backup_providers import BackupProvider
    market_p, stocks_p = patched_paths
    ref_date = datetime(2026, 9, 15, 23, 30)
    write_artifact_with_manifest(
        _make_df(dup_ratio=0.0), str(market_p),
        source='test', reference_date=ref_date, run_id='r_m_nc',
        temporal_contract=None,
    )
    write_artifact_with_manifest(
        _make_df(dup_ratio=0.0), str(stocks_p),
        source='test', reference_date=ref_date, run_id='r_s_nc',
        temporal_contract=None,
    )

    bp = BackupProvider()
    assert bp.reference_cache_status == 'VALID'


def test_contract_equity_eod_future_date_reader_rejects(patched_paths):
    """Con contrato EQUITY_EOD + last_date > expected -> writer INVALID,
    reader INVALID."""
    from data.providers.backup_providers import BackupProvider
    market_p, stocks_p = patched_paths
    ref_date = datetime(2026, 9, 15, 10, 35)
    write_artifact_with_manifest(
        _make_df(last_date='2026-09-15', dup_ratio=0.0), str(market_p),
        source='test', reference_date=ref_date, run_id='r_m_t',
        temporal_contract='EQUITY_EOD',
    )
    write_artifact_with_manifest(
        _make_df(dup_ratio=0.0), str(stocks_p),
        source='test', reference_date=ref_date, run_id='r_s_t',
        temporal_contract='EQUITY_EOD',
    )

    bp = BackupProvider()
    assert bp.reference_cache_status == 'INVALID'


def test_contract_equity_eod_valid_reader_accepts(patched_paths):
    """Con contrato EQUITY_EOD + last_date == expected -> writer VALID,
    reader acepta."""
    from data.providers.backup_providers import BackupProvider
    market_p, stocks_p = patched_paths
    ref_date = datetime(2026, 9, 15, 23, 30)
    write_artifact_with_manifest(
        _make_df(last_date='2026-09-15', dup_ratio=0.0), str(market_p),
        source='test', reference_date=ref_date, run_id='r_m_ok',
        temporal_contract='EQUITY_EOD',
    )
    write_artifact_with_manifest(
        _make_df(dup_ratio=0.0), str(stocks_p),
        source='test', reference_date=ref_date, run_id='r_s_ok',
        temporal_contract='EQUITY_EOD',
    )

    bp = BackupProvider()
    assert bp.reference_cache_status == 'VALID'


def test_schema_v1_with_temporal_block_reader_accepts(patched_paths):
    """Manifest schema_version=1 con bloque temporal: reader acepta.

    Confirma que D2.B (bloque temporal siempre presente) no rompe la FSM
    del reader, que sigue exigiendo schema_version == 1.
    """
    import json as _json
    from data.providers.backup_providers import BackupProvider
    market_p, stocks_p = patched_paths
    ref_date = datetime(2026, 9, 15, 23, 30)
    write_artifact_with_manifest(
        _make_df(dup_ratio=0.0), str(market_p),
        source='test', reference_date=ref_date, run_id='r_m_sch',
        temporal_contract='EQUITY_EOD',
    )
    write_artifact_with_manifest(
        _make_df(dup_ratio=0.0), str(stocks_p),
        source='test', reference_date=ref_date, run_id='r_s_sch',
        temporal_contract='EQUITY_EOD',
    )

    # Verificar que el manifest tiene schema_version=1 y bloque temporal
    mp = Path(str(market_p) + '.manifest.json')
    mdata = _json.loads(mp.read_text(encoding='utf-8'))
    assert mdata['schema_version'] == 1
    assert 'temporal' in mdata
    assert mdata['temporal']['contract'] == 'EQUITY_EOD'

    bp = BackupProvider()
    assert bp.reference_cache_status == 'VALID'

