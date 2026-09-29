"""F5-5 (2026-09-28): ValidationOutcome enum + dedup diagnostic.

Sin red. Se prueban los 6 casos de _validate_with_cache y el
print de solapamiento en _load_reference_cache.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.providers.backup_providers import (
    BackupProvider,
    ValidationOutcome,
)


def _make_cache_df(tickers, close_value=100.0, n_rows=10):
    dates = pd.date_range("2026-01-01", periods=n_rows, freq="D")
    cols = pd.MultiIndex.from_product([["Close"], tickers])
    return pd.DataFrame(close_value, index=dates, columns=cols)


def _make_provider_df(ticker, close_value):
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    cols = pd.MultiIndex.from_product([["Close"], [ticker]])
    return pd.DataFrame(close_value, index=dates, columns=cols)


@pytest.fixture
def bp_empty():
    """BackupProvider con cache deshabilitada (evita dependencia de disco)."""
    bp = BackupProvider.__new__(BackupProvider)
    bp.reference_cache = pd.DataFrame()
    bp.reference_cache_status = 'VALID'
    return bp


def test_outcome_unavailable(bp_empty):
    bp_empty.reference_cache_status = 'UNAVAILABLE'
    out = bp_empty._validate_with_cache('AAPL', _make_provider_df('AAPL', 100.0))
    assert out is ValidationOutcome.UNAVAILABLE


def test_outcome_cache_empty(bp_empty):
    out = bp_empty._validate_with_cache('AAPL', _make_provider_df('AAPL', 100.0))
    assert out is ValidationOutcome.CACHE_EMPTY


def test_outcome_ticker_not_in_cache(bp_empty):
    bp_empty.reference_cache = _make_cache_df(['MSFT'])
    out = bp_empty._validate_with_cache('AAPL', _make_provider_df('AAPL', 100.0))
    assert out is ValidationOutcome.TICKER_NOT_IN_CACHE


def test_outcome_validated(bp_empty):
    bp_empty.reference_cache = _make_cache_df(['AAPL'], close_value=100.0)
    out = bp_empty._validate_with_cache('AAPL', _make_provider_df('AAPL', 101.0))
    assert out is ValidationOutcome.VALIDATED


def test_outcome_rejected(bp_empty):
    bp_empty.reference_cache = _make_cache_df(['AAPL'], close_value=100.0)
    out = bp_empty._validate_with_cache('AAPL', _make_provider_df('AAPL', 120.0))
    assert out is ValidationOutcome.REJECTED


def test_outcome_insufficient_data_ref_close_zero(bp_empty):
    bp_empty.reference_cache = _make_cache_df(['AAPL'], close_value=0.0)
    out = bp_empty._validate_with_cache('AAPL', _make_provider_df('AAPL', 100.0))
    assert out is ValidationOutcome.INSUFFICIENT_DATA


def test_outcome_enum_tiene_7_valores():
    assert len(list(ValidationOutcome)) == 7
    assert {o.name for o in ValidationOutcome} == {
        'VALIDATED', 'REJECTED', 'UNAVAILABLE', 'CACHE_EMPTY',
        'TICKER_NOT_IN_CACHE', 'INSUFFICIENT_DATA', 'ERROR'
    }

# --- Fix 2026-09-29: finitud en _validate_with_cache ---------------------

def test_outcome_insufficient_data_new_close_nan(bp_empty):
    """Bug fix 2026-09-29: NaN en new_close pasaba como VALIDATED
    porque nan > 0.05 es False y no entraba en la rama REJECTED."""
    bp_empty.reference_cache = _make_cache_df(['AAPL'], close_value=100.0)
    df = _make_provider_df('AAPL', 100.0)
    # ultima fila NaN
    df.iloc[-1, 0] = float('nan')
    out = bp_empty._validate_with_cache('AAPL', df)
    assert out is ValidationOutcome.INSUFFICIENT_DATA


def test_outcome_insufficient_data_new_close_inf(bp_empty):
    bp_empty.reference_cache = _make_cache_df(['AAPL'], close_value=100.0)
    df = _make_provider_df('AAPL', 100.0)
    df.iloc[-1, 0] = float('inf')
    out = bp_empty._validate_with_cache('AAPL', df)
    assert out is ValidationOutcome.INSUFFICIENT_DATA


def test_outcome_insufficient_data_new_close_neg_inf(bp_empty):
    bp_empty.reference_cache = _make_cache_df(['AAPL'], close_value=100.0)
    df = _make_provider_df('AAPL', 100.0)
    df.iloc[-1, 0] = float('-inf')
    out = bp_empty._validate_with_cache('AAPL', df)
    assert out is ValidationOutcome.INSUFFICIENT_DATA


def test_outcome_insufficient_data_ref_close_inf(bp_empty):
    """ref_close viene de .dropna() que NO elimina inf.
    Cualquier inf en cache invalida la comparacion."""
    bp_empty.reference_cache = _make_cache_df(['AAPL'], close_value=100.0)
    # inf en la ultima fila del cache, dropna no lo elimina
    bp_empty.reference_cache.iloc[-1, 0] = float('inf')
    out = bp_empty._validate_with_cache('AAPL', _make_provider_df('AAPL', 100.0))
    assert out is ValidationOutcome.INSUFFICIENT_DATA