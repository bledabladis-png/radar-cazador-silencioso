"""Fixtures compartidas de tests.

FU-021-5-fix-ci: los tests de contratos temporales dependen del parquet
real (data/market_data.parquet), gitignored en el repo (~57 MB). En CI
no existe. La fixture `df_real` skipea elegantemente en ese caso.
"""
from pathlib import Path

import pandas as pd
import pytest


_MARKET_PARQUET = Path("data/market_data.parquet")
_STOCK_PARQUET = Path("data/stock_prices.parquet")


@pytest.fixture
def df_real():
    """df_market real desde parquet local. Skip si no disponible (CI)."""
    if not _MARKET_PARQUET.exists():
        pytest.skip(
            f"CI: {_MARKET_PARQUET} no disponible (gitignored). "
            "Tests que dependen de datos reales se omiten."
        )
    return pd.read_parquet(_MARKET_PARQUET)


@pytest.fixture
def df_stocks_real():
    """df_stocks real desde parquet local. Skip si no disponible (CI)."""
    if not _STOCK_PARQUET.exists():
        pytest.skip(
            f"CI: {_STOCK_PARQUET} no disponible (gitignored)."
        )
    return pd.read_parquet(_STOCK_PARQUET)