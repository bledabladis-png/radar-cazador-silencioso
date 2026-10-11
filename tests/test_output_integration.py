# -*- coding: utf-8 -*-
"""Test de integracion: el flujo collector->verify cubre el gap 09-oct."""
import pandas as pd

from src.pipeline.output_registry import REGISTRY, ReceiptCollector, WriteReceipt
from src.pipeline.output_verifier import verify


def _seed_all(root, date='2026-10-10'):
    (root / 'outputs/history').mkdir(parents=True, exist_ok=True)
    for spec in REGISTRY:
        if spec.kind == 'temporal':
            pd.DataFrame({'date': [date], 'sector': ['XLK']}).to_csv(
                root / spec.path, index=False)
        else:
            pd.DataFrame({'sector': ['XLK']}).to_csv(
                root / spec.path, index=False)


def _emit_all(c, date='2026-10-10'):
    for spec in REGISTRY:
        if spec.kind == 'temporal':
            c.add(WriteReceipt(spec.name, True, 'OK', date, spec.path, 1))
        else:
            c.add(WriteReceipt(spec.name, True, 'OK', None, spec.path, 1))


def test_full_flow_all_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed_all(tmp_path)
    c = ReceiptCollector()
    _emit_all(c)
    result = verify(c)
    assert result['passed'] is True, result['errors']
    assert len(result['checks']) == len(REGISTRY)


def test_full_flow_silent_writer_reproduces_gap_09oct(tmp_path, monkeypatch):
    """Reproduce el gap: sector_breadth no emite receipt. El verificador
    lo caza por R1 sin depender de fechas."""
    monkeypatch.chdir(tmp_path)
    _seed_all(tmp_path)
    c = ReceiptCollector()
    for spec in REGISTRY:
        if spec.name == 'sector_breadth':
            continue  # <-- writer silencioso
        if spec.kind == 'temporal':
            c.add(WriteReceipt(spec.name, True, 'OK', '2026-10-10', spec.path, 1))
        else:
            c.add(WriteReceipt(spec.name, True, 'OK', None, spec.path, 1))
    result = verify(c)
    assert result['passed'] is False
    assert any('sector_breadth' in e and 'sin receipt' in e
               for e in result['errors'])


def test_full_flow_stale_writer_passes_with_skip(tmp_path, monkeypatch):
    """Un writer que salta legitimamente (STALE) pasa la verificacion
    si el CSV existe y no esta vacio."""
    monkeypatch.chdir(tmp_path)
    _seed_all(tmp_path)
    c = ReceiptCollector()
    for spec in REGISTRY:
        if spec.name == 'sector_breadth':
            c.add(WriteReceipt('sector_breadth', False, 'STALE', None,
                               spec.path, 0))
        elif spec.kind == 'temporal':
            c.add(WriteReceipt(spec.name, True, 'OK', '2026-10-10', spec.path, 1))
        else:
            c.add(WriteReceipt(spec.name, True, 'OK', None, spec.path, 1))
    result = verify(c)
    assert result['passed'] is True, result['errors']
