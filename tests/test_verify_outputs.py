# -*- coding: utf-8 -*-
"""Tests del verificador extendido de outputs."""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_outputs import verify_historical, _expected_sessions


def _seed_csv(root, spec, dates):
    p = root / spec.path
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({'date': [str(d) for d in dates]})
    if 'sector' in spec.merge_keys:
        df['sector'] = 'XLK'
    if 'source' in spec.merge_keys:
        df['source'] = 'yahoo'
    df.to_csv(p, index=False)


def test_verify_all_ok(tmp_path, monkeypatch):
    """Rellena TODOS los outputs con las fechas que espera el verificador."""
    monkeypatch.setattr('scripts.verify_outputs.PROJECT_ROOT', tmp_path)
    # Importante: monkeypatchar el path de cada spec
    from src.pipeline.output_registry import OutputSpec
    fake_registry = []
    ref = date(2026, 10, 12)
    since = date(2026, 10, 1)
    # Usar una ventana de 3 y fabricar CSVs acordes
    for spec in [s for s in __import__('src.pipeline.output_registry',
                                        fromlist=['REGISTRY']).REGISTRY
                 if s.kind == 'temporal' and s.name != 'data_quality']:
        sessions = _expected_sessions(ref, 3, since)
        fake_spec = OutputSpec(
            spec.name, spec.path, spec.kind, spec.date_column, spec.merge_keys,
        )
        _seed_csv(tmp_path, fake_spec, sessions)
        fake_registry.append(fake_spec)
    results = verify_historical(3, ref, since, registry=fake_registry)
    gaps = [r for r in results if r['status'] != 'OK']
    assert not gaps, gaps


def test_verify_detects_gap(tmp_path, monkeypatch):
    """Quita una fecha y el verificador la caza."""
    monkeypatch.setattr('scripts.verify_outputs.PROJECT_ROOT', tmp_path)
    from src.pipeline.output_registry import OutputSpec
    ref = date(2026, 10, 12)
    since = date(2026, 10, 1)
    sessions = _expected_sessions(ref, 3, since)
    spec = OutputSpec('sector_breadth',
                      'outputs/history/sector_breadth.csv',
                      'temporal', 'date', ('date', 'sector'))
    _seed_csv(tmp_path, spec, sessions[:-1])  # falta la ultima
    results = verify_historical(3, ref, since, registry=[spec])
    assert results[0]['status'] == 'GAP'
    assert len(results[0]['missing']) == 1


def test_verify_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.verify_outputs.PROJECT_ROOT', tmp_path)
    from src.pipeline.output_registry import OutputSpec
    ref = date(2026, 10, 12)
    since = date(2026, 10, 1)
    spec = OutputSpec('sector_breadth',
                      'outputs/history/sector_breadth.csv',
                      'temporal', 'date', ('date', 'sector'))
    results = verify_historical(3, ref, since, registry=[spec])
    assert results[0]['status'] == 'MISSING_FILE'


def test_expected_sessions_respects_since():
    """since corta la ventana."""
    ref = date(2026, 10, 12)
    since = date(2026, 10, 8)
    out = _expected_sessions(ref, 10, since)
    assert all(d >= since for d in out)
    assert len(out) < 10


def test_verify_upstream_output_only_checks_presence(tmp_path, monkeypatch):
    """Outputs con date_source='upstream' no se verifican por fecha de
    sesion: solo se comprueba que el CSV exista y no este vacio."""
    from src.pipeline.output_registry import OutputSpec
    monkeypatch.setattr('scripts.verify_outputs.PROJECT_ROOT', tmp_path)
    ref = date(2026, 10, 12)
    since = date(2026, 10, 1)
    spec = OutputSpec(
        'sector_flow_characteristics',
        'outputs/history/sector_flow_characteristics.csv',
        'temporal', 'date', ('date', 'sector'),
        date_source='upstream',
    )
    # CSV con fecha antigua (upstream lag)
    p = tmp_path / spec.path
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'date': ['2026-10-09'], 'sector': ['XLK']}).to_csv(p, index=False)
    results = verify_historical(3, ref, since, registry=[spec])
    assert results[0]['status'] == 'OK_UPSTREAM'
    assert results[0]['missing'] == []


def test_verify_upstream_output_missing_file(tmp_path, monkeypatch):
    """Output upstream sin fichero -> MISSING_FILE."""
    from src.pipeline.output_registry import OutputSpec
    monkeypatch.setattr('scripts.verify_outputs.PROJECT_ROOT', tmp_path)
    ref = date(2026, 10, 12)
    since = date(2026, 10, 1)
    spec = OutputSpec(
        'sector_flow_characteristics',
        'outputs/history/sector_flow_characteristics.csv',
        'temporal', 'date', ('date', 'sector'),
        date_source='upstream',
    )
    results = verify_historical(3, ref, since, registry=[spec])
    assert results[0]['status'] == 'MISSING_FILE'
