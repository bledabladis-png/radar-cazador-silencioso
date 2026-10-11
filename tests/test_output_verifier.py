# -*- coding: utf-8 -*-
"""Tests del verificador post-run."""
import pandas as pd
import pytest

from src.pipeline.output_registry import REGISTRY, ReceiptCollector, WriteReceipt
from src.pipeline.output_verifier import verify


def _seed_all_csvs(root, date='2026-10-10'):
    (root / 'outputs/history').mkdir(parents=True, exist_ok=True)
    for spec in REGISTRY:
        if spec.kind == 'temporal':
            pd.DataFrame({'date': [date], 'sector': ['XLK']}).to_csv(
                root / spec.path, index=False)
        else:
            pd.DataFrame({'sector': ['XLK']}).to_csv(
                root / spec.path, index=False)


def test_verify_no_receipts_all_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed_all_csvs(tmp_path)
    result = verify(ReceiptCollector())
    assert result['passed'] is False
    assert len(result['errors']) == len(REGISTRY)


def test_verify_silent_writer_caught(tmp_path, monkeypatch):
    """Caso 09-oct: un writer no emite receipt. El verificador lo caza
    por R1 (receipt ausente), no por fallo de fecha."""
    monkeypatch.chdir(tmp_path)
    _seed_all_csvs(tmp_path)
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


def test_verify_written_but_csv_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'outputs/history').mkdir(parents=True)
    c = ReceiptCollector()
    c.add(WriteReceipt('sector_breadth', True, 'OK', '2026-10-10',
                       'outputs/history/sector_breadth.csv', 1))
    result = verify(c)
    assert result['passed'] is False
    assert any('no existe' in e and 'sector_breadth' in e
               for e in result['errors'])


def test_verify_written_but_date_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'outputs/history').mkdir(parents=True)
    pd.DataFrame({'date': ['2026-10-08'], 'sector': ['XLK']}).to_csv(
        tmp_path / 'outputs/history/sector_breadth.csv', index=False)
    c = ReceiptCollector()
    c.add(WriteReceipt('sector_breadth', True, 'OK', '2026-10-10',
                       'outputs/history/sector_breadth.csv', 1))
    result = verify(c)
    assert result['passed'] is False
    assert any('no tiene esa fecha' in e for e in result['errors'])


def test_verify_skip_illegitimate_reason_caught(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'outputs/history').mkdir(parents=True)
    pd.DataFrame({'date': ['2026-10-09'], 'sector': ['XLK']}).to_csv(
        tmp_path / 'outputs/history/sector_breadth.csv', index=False)
    c = ReceiptCollector()
    c.add(WriteReceipt('sector_breadth', False, 'PORQUE_SI', None,
                       'outputs/history/sector_breadth.csv', 0))
    result = verify(c)
    assert result['passed'] is False
    assert any('no legitima' in e for e in result['errors'])


def test_verify_skip_but_csv_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'outputs/history').mkdir(parents=True)
    pd.DataFrame({'date': [], 'sector': []}).to_csv(
        tmp_path / 'outputs/history/sector_breadth.csv', index=False)
    c = ReceiptCollector()
    c.add(WriteReceipt('sector_breadth', False, 'STALE', None,
                       'outputs/history/sector_breadth.csv', 0))
    result = verify(c)
    assert result['passed'] is False
    assert any('esta vacio' in e for e in result['errors'])


def test_verify_all_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed_all_csvs(tmp_path)
    c = ReceiptCollector()
    for spec in REGISTRY:
        if spec.kind == 'temporal':
            c.add(WriteReceipt(spec.name, True, 'OK', '2026-10-10',
                               spec.path, 1))
        else:
            c.add(WriteReceipt(spec.name, True, 'OK', None, spec.path, 1))
    result = verify(c)
    assert result['passed'] is True, result['errors']


def test_verify_snapshot_kind_no_date_check(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed_all_csvs(tmp_path)
    c = ReceiptCollector()
    for spec in REGISTRY:
        if spec.kind == 'temporal':
            c.add(WriteReceipt(spec.name, True, 'OK', '2026-10-10',
                               spec.path, 1))
        else:
            c.add(WriteReceipt(spec.name, True, 'OK', None, spec.path, 1))
    result = verify(c)
    assert result['passed'] is True, result['errors']
