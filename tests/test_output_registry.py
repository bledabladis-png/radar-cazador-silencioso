# -*- coding: utf-8 -*-
"""Tests del registro declarativo de outputs."""
import pytest

from src.pipeline.output_registry import (
    REGISTRY, get_spec, WriteReceipt, ReceiptCollector,
    LEGITIMATE_SKIP_REASONS,
)


def test_registry_count():
    assert len(REGISTRY) == 19


def test_registry_unique_names():
    names = [s.name for s in REGISTRY]
    assert len(names) == len(set(names))


def test_registry_kinds_valid():
    for s in REGISTRY:
        assert s.kind in ('temporal', 'snapshot')


def test_registry_temporal_has_date_and_merge_keys():
    for s in REGISTRY:
        if s.kind == 'temporal':
            assert s.date_column == 'date'
            assert len(s.merge_keys) > 0


def test_registry_snapshot_no_date_no_merge():
    for s in REGISTRY:
        if s.kind == 'snapshot':
            assert s.date_column == ''
            assert s.merge_keys == ()


def test_sector_rank_deltas_is_snapshot():
    spec = get_spec('sector_rank_deltas')
    assert spec.kind == 'snapshot'


def test_get_spec_unknown_raises():
    with pytest.raises(KeyError):
        get_spec('no_existe')


def test_legitimate_skip_reasons_frozen():
    for r in ('STALE', 'NO_DATA', 'SKIP_CONDITION', 'EMPTY_DF'):
        assert r in LEGITIMATE_SKIP_REASONS


def test_receipt_to_dict():
    r = WriteReceipt('sector_breadth', True, 'OK', '2026-10-10',
                     'outputs/history/sector_breadth.csv', 22)
    d = r.to_dict()
    assert d['name'] == 'sector_breadth'
    assert d['written'] is True
    assert d['reason'] == 'OK'
    assert d['date'] == '2026-10-10'
    assert d['rows'] == 22


def test_receipt_to_dict_none_date():
    r = WriteReceipt('x', False, 'STALE', None, 'p', 0)
    assert r.to_dict()['date'] is None


def test_collector_by_name():
    c = ReceiptCollector()
    c.add(WriteReceipt('a', True, 'OK', None, 'p', 0))
    c.add(WriteReceipt('b', False, 'STALE', None, 'q', 0))
    bn = c.by_name()
    assert bn['a'].written is True
    assert bn['b'].written is False


def test_collector_to_dict():
    c = ReceiptCollector()
    c.add(WriteReceipt('a', True, 'OK', '2026-10-10', 'p', 1))
    d = c.to_dict()
    assert d['schema_version'] == 1
    assert len(d['receipts']) == 1
