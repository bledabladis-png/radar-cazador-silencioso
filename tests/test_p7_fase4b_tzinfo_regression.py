"""Regresion a6a8f017: preservar tz-aware en orquestadores.

El naive que devuelve resolve_observation_date no debe viajar a
is_session_closed, que exige tz-aware (raise duro). Detectado en run
40787015709 (workflow_dispatch 28-10 23:23Z, failure a los 8m).
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest


def _fake_router_response(batch):
    cols = pd.MultiIndex.from_tuples(
        [(m, t) for m in ('Open', 'High', 'Low', 'Close', 'Volume') for t in batch]
    )
    idx = pd.DatetimeIndex([pd.Timestamp('2026-10-28')])
    return pd.DataFrame(1.0, index=idx, columns=cols)


def test_download_market_data_preserves_tz_aware(monkeypatch, tmp_path):
    from src import data_loader
    captured = {}

    def _spy(data, reference_date, run_id, *, write_manifest):
        captured['ref'] = reference_date
        return data

    monkeypatch.setattr(data_loader, '_postprocess_market_data', _spy)

    class _FakeRouter:
        def get_market_data(self, batch, period='10y'):
            return _fake_router_response(batch)

    monkeypatch.setattr(data_loader, 'DataRouter', _FakeRouter)
    monkeypatch.setattr(data_loader, '_ticker_list', lambda: ['AAPL'])
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'data').mkdir()

    ref = datetime(2026, 10, 28, 23, 0, tzinfo=ZoneInfo('Europe/Madrid'))
    data_loader.download_market_data(reference_date=ref, run_id='test')

    assert captured['ref'].tzinfo is not None, (
        'download_market_data muto reference_date a naive (regresion a6a8f017)'
    )


def test_download_market_data_raises_on_none(monkeypatch, tmp_path):
    from src import data_loader
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'data').mkdir()
    with pytest.raises(ValueError, match='reference_date=None'):
        data_loader.download_market_data(reference_date=None, run_id='test')


def test_resolve_observation_date_no_muta_parametro():
    from src.utils import resolve_observation_date
    ref = datetime(2026, 10, 28, tzinfo=ZoneInfo('Europe/Madrid'))
    out = resolve_observation_date(ref)
    assert out.tzinfo is None
    assert ref.tzinfo is not None


def test_no_reasignacion_parametro_con_resolve_observation_date():
    """Guard AST: prohibido `reference_date = resolve_observation_date(...)`."""
    import ast
    from pathlib import Path
    from tests.test_fase4_observacion_vs_formato import OBSERVACION_MODULOS

    ROOT = Path(__file__).resolve().parents[1]
    for rel, func in OBSERVACION_MODULOS:
        src = (ROOT / rel).read_text(encoding='utf-8-sig', errors='replace')
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name == func):
                continue
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Assign):
                    continue
                targets = [t.id for t in sub.targets if isinstance(t, ast.Name)]
                if 'reference_date' not in targets:
                    continue
                rhs = sub.value
                if (isinstance(rhs, ast.Call)
                        and isinstance(rhs.func, ast.Name)
                        and rhs.func.id == 'resolve_observation_date'):
                    pytest.fail(
                        f'{rel}::{func}: prohibido reference_date = '
                        f'resolve_observation_date(...). Usar `_ = ...`.'
                    )
