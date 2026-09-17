"""K-DATA-LOADER-01 (2026-09-17): tests del post-procesado comun.

Verifica que:
  - write_manifest=True escribe parquet/manifest (cache-miss).
  - write_manifest=False NO escribe parquet (cache-hit).
  - Ambos caminos setean data.attrs['temporal_meta'].
  - download_market_data en cache-hit aplica post-procesado.
"""
import importlib
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import src.data_loader as dl

# Cargar modulos via importlib. Los imports directos (src.temporal_contracts.consolidate)
# fallan porque __init__.py sombrea el nombre del submodulo con un atributo funcion.
_cmerge = importlib.import_module('src.commodities_merge')
_cboemerge = importlib.import_module('src.cboe_merge')
_utils = importlib.import_module('src.utils')
_tc = importlib.import_module('src.temporal_contracts')
_tc_consolidate = importlib.import_module('src.temporal_contracts.consolidate')


def _make_df(n_days=5, n_tickers=3):
    """df minimo con MultiIndex de columnas (field, ticker)."""
    dates = pd.date_range('2026-09-01', periods=n_days, freq='B')
    fields = ['Close', 'Open', 'High', 'Low', 'Volume']
    tickers = [f'TICK{i}' for i in range(n_tickers)]
    cols = pd.MultiIndex.from_product([fields, tickers], names=['field', 'ticker'])
    rng = np.random.RandomState(42)
    data = rng.rand(n_days, len(cols)) + 100.0
    return pd.DataFrame(data, index=dates, columns=cols)


@pytest.fixture
def patched_pipeline(monkeypatch):
    """Aisla _postprocess_market_data de merges, filter EOD y contracts reales."""
    calls = {'writer': [], 'merge_comm': 0, 'merge_cboe': 0}

    monkeypatch.setattr(dl, '_filter_non_eod_equity', lambda d, r: (d, {}))
    monkeypatch.setattr(dl, '_trim_market_data_to_equity_eod', lambda d: (d, None))

    def _merge_comm(d):
        calls['merge_comm'] += 1
        return d

    def _merge_cboe(d):
        calls['merge_cboe'] += 1
        return d

    monkeypatch.setattr(_cmerge, 'merge_commodities_into_market', _merge_comm)
    monkeypatch.setattr(_cboemerge, 'merge_cboe_into_market', _merge_cboe)

    def _writer(d, path, source, reference_date, run_id):
        calls['writer'].append({'path': path, 'source': source, 'run_id': run_id})

    monkeypatch.setattr(_utils, 'write_artifact_with_manifest', _writer)

    class _R:
        status = 'OK'

    def _resolve(data, ref):
        return {'EQUITY_EOD': _R()}

    def _build(res, ref, rid):
        return {'stub': True, 'run_id': rid}

    monkeypatch.setattr(_tc, 'resolve_all_contracts', _resolve)
    monkeypatch.setattr(_tc_consolidate, 'build_temporal_meta', _build)

    return calls


_REF = datetime(2026, 9, 5, 20, 0, tzinfo=ZoneInfo('Europe/Madrid'))


def test_postprocess_write_manifest_true_calls_writer(patched_pipeline):
    df = _make_df()
    dl._postprocess_market_data(df, _REF, 'rid_001', write_manifest=True)
    assert len(patched_pipeline['writer']) == 1
    w = patched_pipeline['writer'][0]
    assert w['path'] == 'data/market_data.parquet'
    assert w['source'] == 'yahoo'
    assert w['run_id'] == 'rid_001'
    assert patched_pipeline['merge_comm'] == 1
    assert patched_pipeline['merge_cboe'] == 1


def test_postprocess_write_manifest_false_does_not_call_writer(patched_pipeline):
    df = _make_df()
    dl._postprocess_market_data(df, _REF, 'rid_002', write_manifest=False)
    assert patched_pipeline['writer'] == []
    assert patched_pipeline['merge_comm'] == 1
    assert patched_pipeline['merge_cboe'] == 1


def test_postprocess_sets_temporal_meta_both_modes(patched_pipeline):
    for mode in (True, False):
        df = _make_df()
        out = dl._postprocess_market_data(df, _REF, f'rid_{mode}', write_manifest=mode)
        meta = out.attrs.get('temporal_meta')
        assert meta is not None, f'write_manifest={mode}'
        assert meta.get('stub') is True
        assert meta.get('run_id') == f'rid_{mode}'


def test_download_cache_hit_applies_postprocess(monkeypatch, tmp_path, patched_pipeline):
    """Cache-hit NO debe reescribir el parquet, pero SI setear temporal_meta."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'data').mkdir()
    df = _make_df()
    df.to_parquet(tmp_path / 'data' / 'market_data.parquet')

    monkeypatch.setattr(dl, 'CACHE_VALIDATE_TRADING_DATE', False)

    ref = datetime.now(ZoneInfo('Europe/Madrid'))
    out = dl.download_market_data(reference_date=ref, run_id='rid_cache_hit')

    assert out is not None
    assert isinstance(out, pd.DataFrame)
    assert out.attrs.get('temporal_meta') is not None
    assert out.attrs['temporal_meta'].get('run_id') == 'rid_cache_hit'
    assert patched_pipeline['writer'] == []