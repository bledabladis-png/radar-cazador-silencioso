# -*- coding: utf-8 -*-
"""Tests del registro declarativo de outputs."""
import pytest

from src.pipeline.output_registry import (
    REGISTRY, get_spec, WriteReceipt, ReceiptCollector,
    LEGITIMATE_SKIP_REASONS,
)


def test_registry_count():
    assert len(REGISTRY) == 26


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

# ============================================================
# N7 (2026-10-13): coherencia merge_keys vs columnas reales.
# Bug detectado: evidence_matrix y sector_regime_matrix tenian
# merge_keys=('date',) cuando el CSV tiene columna sector.
# Consecuencia: append_dedup solo conservaba 1 fila por fecha
# (se quedaba con el ultimo sector). Bug pre-existente a la
# migracion del contrato de persistencia. Fix 2026-10-13.
# ============================================================

def test_merge_keys_incluye_sector_si_csv_lo_tiene():
    """Cualquier OutputSpec temporal cuyo CSV tenga columna sector
    debe declarar sector en merge_keys. Previene la clase de bug
    N7 (perdida de 10 de 11 sectores al deduplicar)."""
    import os
    import pandas as pd
    from src.pipeline.output_registry import REGISTRY

    issues = []
    for spec in REGISTRY:
        if spec.kind != 'temporal':
            continue
        if not os.path.exists(spec.path):
            continue
        try:
            cols = pd.read_csv(spec.path, nrows=0).columns.tolist()
        except Exception:
            continue
        if 'sector' in cols and 'sector' not in spec.merge_keys:
            issues.append(
                f"{spec.name}: CSV tiene 'sector' pero merge_keys="
                f"{spec.merge_keys}"
            )
    assert not issues, (
        "Outputs con columna sector sin sector en merge_keys:\n  - "
        + "\n  - ".join(issues)
    )
