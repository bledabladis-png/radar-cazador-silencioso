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


_TRACKED_SCHEMAS_CACHE = None


def _get_tracked_schemas():
    """Schemas de outputs/history/ trackeados en git (cache por sesion)."""
    global _TRACKED_SCHEMAS_CACHE
    if _TRACKED_SCHEMAS_CACHE is None:
        import subprocess as _sp
        r = _sp.run(
            ["git", "ls-files", "outputs/history/*.schema"],
            capture_output=True, text=True, check=False,
        )
        _TRACKED_SCHEMAS_CACHE = set(
            line.replace("\\", "/").strip()
            for line in r.stdout.splitlines() if line.strip()
        )
    return _TRACKED_SCHEMAS_CACHE


@pytest.fixture(autouse=True)
def _cleanup_schema_files():
    """Limpia/restaura .schema modificados por el test en outputs/history/.

    Contrato (revisado 2026-10-29):
      - mtime < start -> no tocar (produccion antigua intacta).
      - tracked en git -> `git checkout --` restaura version del bot.
      - untracked -> unlink() (residuo real de test).

    Incidente 2026-10-25: version previa borraba TODOS los .schema
    de outputs/history/, rompia sector_wyckoff_distribution.

    Incidente 2026-10-29: filtro por mtime dejaba pasar
    macro_regime.csv.schema (tracked, reescrito por test).
    unlink -> D en git status tras la suite. Fix: tracked ->
    checkout, untracked -> unlink.
    """
    import subprocess as _sp
    import time as _time
    _start = _time.time()
    yield
    _history = Path(__file__).parent.parent / "outputs" / "history"
    if not _history.exists():
        return
    tracked = _get_tracked_schemas()
    for _schema in _history.glob("*.schema"):
        try:
            if _schema.stat().st_mtime < _start:
                continue
            _rel = str(_schema.relative_to(_history.parent.parent)).replace("\\", "/")
            if _rel in tracked:
                _sp.run(["git", "checkout", "--", _rel], check=False, capture_output=True)
            else:
                _schema.unlink()
        except OSError:
            pass
