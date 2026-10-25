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


@pytest.fixture(autouse=True)
def _cleanup_schema_files():
    """Limpia .schema generados por tests que escriben al path real.

    `save_regime_history` escribe `outputs/history/macro_regime.csv.schema`
    como side-effect cuando el test no pasa `path_override`. El .schema es
    metadata local (gitignored) y no debe sobrevivir entre tests ni tras
    la suite. No se borra el CSV (forma parte del repo).
    """
    yield
    _history = Path(__file__).parent.parent / "outputs" / "history"
    if _history.exists():
        for _schema in _history.glob("*.schema"):
            try:
                _schema.unlink()
            except OSError:
                pass
