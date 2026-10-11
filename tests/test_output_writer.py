# -*- coding: utf-8 -*-
"""Tests del wrapper de persistencia."""
import pandas as pd
import pytest

from src.pipeline.output_writer import persist, skip
from src.pipeline.output_registry import ReceiptCollector


def test_persist_new_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({'date': ['2026-10-10'], 'sector': ['XLK']})
    r = persist('sector_breadth', df, date_value='2026-10-10')
    assert r.written is True
    assert r.reason == 'OK'
    assert (tmp_path / 'outputs' / 'history' / 'sector_breadth.csv').exists()


def test_persist_merges_with_existing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'outputs' / 'history').mkdir(parents=True)
    pd.DataFrame({'date': ['2026-10-09'], 'sector': ['XLK']}).to_csv(
        tmp_path / 'outputs/history/sector_breadth.csv', index=False)
    df = pd.DataFrame({'date': ['2026-10-10'], 'sector': ['XLK']})
    r = persist('sector_breadth', df, date_value='2026-10-10')
    assert r.written is True
    out = pd.read_csv(tmp_path / 'outputs/history/sector_breadth.csv')
    assert len(out) == 2


def test_persist_dedup_by_merge_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'outputs' / 'history').mkdir(parents=True)
    pd.DataFrame({'date': ['2026-10-10'], 'sector': ['XLK'], 'v': [1]}).to_csv(
        tmp_path / 'outputs/history/sector_breadth.csv', index=False)
    df = pd.DataFrame({'date': ['2026-10-10'], 'sector': ['XLK'], 'v': [2]})
    persist('sector_breadth', df, date_value='2026-10-10')
    out = pd.read_csv(tmp_path / 'outputs/history/sector_breadth.csv')
    assert len(out) == 1
    assert out['v'].iloc[0] == 2


def test_persist_empty_df_returns_skip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = persist('sector_breadth', pd.DataFrame(), date_value='2026-10-10')
    assert r.written is False
    assert r.reason == 'EMPTY_DF'


def test_persist_none_df_returns_skip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = persist('sector_breadth', None)
    assert r.written is False
    assert r.reason == 'EMPTY_DF'


def test_persist_unknown_name_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(KeyError):
        persist('no_existe', pd.DataFrame({'date': ['2026-10-10']}))


def test_persist_missing_date_column_returns_no_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({'sector': ['XLK']})
    r = persist('sector_breadth', df, date_value='2026-10-10')
    assert r.written is False
    assert r.reason == 'NO_DATA'


def test_persist_snapshot_no_merge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({'sector': ['XLK'], 'rank_actual': [1]})
    r = persist('sector_rank_deltas', df)
    assert r.written is True


def test_persist_with_collector(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = ReceiptCollector()
    df = pd.DataFrame({'date': ['2026-10-10'], 'sector': ['XLK']})
    persist('sector_breadth', df, date_value='2026-10-10', collector=c)
    assert len(c.receipts()) == 1
    assert c.receipts()[0].name == 'sector_breadth'


def test_skip_legitimate_reason(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = skip('sector_breadth', 'STALE')
    assert r.written is False
    assert r.reason == 'STALE'


def test_skip_illegitimate_reason_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        skip('sector_breadth', 'PORQUE_SI')


def test_skip_unknown_name_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(KeyError):
        skip('no_existe', 'STALE')


def test_skip_with_collector(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = ReceiptCollector()
    skip('evidence_matrix', 'NO_DATA', collector=c)
    assert len(c.receipts()) == 1
    assert c.receipts()[0].written is False
