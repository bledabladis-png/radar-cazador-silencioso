"""Tests de cache freshness (cache-audit 2026-10-28).

3 hallazgos del audit:

1. data_loader.py cache-hit no validaba que el parquet tuviera todos
   los tickers de _ticker_list(). Si se anade un ETF a etf_holdings.csv,
   el cache lo omite hasta que caduque por mtime. stock_data_loader.py
   ya lo validaba; ahora ambos.

2. CSV legacy (market_data.csv, stock_prices.csv) nunca se escriben por
   codigo pero el cache-hit los preferia por mtime. Un touch manual
   resucitaba datos obsoletos. Eliminada la rama CSV.

3. min_coverage=0.90 hardcoded en 2 loaders + 1 llamada redundante en
   stocks_data.py. Extraido a CACHE_MIN_COVERAGE en config.settings
   para los loaders; eliminado el kwarg redundante en stocks_data
   (el default de resolve_effective_date ya es 0.90).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import CACHE_MIN_COVERAGE
from src.utils import _last_row_coverage_ok


def _df_multiindex(tickers, n=3, close_nan_last=False):
    idx = pd.DatetimeIndex(["2026-10-23", "2026-10-26", "2026-10-27"])
    fields = ["Open", "High", "Low", "Close", "Volume"]
    cols = pd.MultiIndex.from_product([fields, tickers], names=["field", "ticker"])
    data = np.ones((n, len(cols)))
    if close_nan_last:
        for i, c in enumerate(cols):
            if c[0] == "Close":
                data[-1, i] = np.nan
    return pd.DataFrame(data, index=idx[:n], columns=cols)


def test_cache_min_coverage_config():
    assert isinstance(CACHE_MIN_COVERAGE, float)
    assert 0.0 < CACHE_MIN_COVERAGE <= 1.0


def test_last_row_coverage_ok_acepta_min_coverage():
    df_ok = _df_multiindex(["AAA", "BBB"], close_nan_last=False)
    assert _last_row_coverage_ok(df_ok, min_coverage=CACHE_MIN_COVERAGE) is True
    df_nan = _df_multiindex(["AAA", "BBB"], close_nan_last=True)
    assert _last_row_coverage_ok(df_nan, min_coverage=CACHE_MIN_COVERAGE) is False


def test_data_loader_valida_tickers_en_cache_hit():
    """AST: data_loader.py contiene el check _missing (tickers de
    _ticker_list() vs _cached_tickers)."""
    src = (ROOT / "src" / "data_loader.py").read_text(encoding="utf-8-sig", errors="replace")
    assert "_missing = set(_ticker_list()) - _cached_tickers" in src, (
        "data_loader.py no valida que el parquet cacheado tenga todos "
        "los tickers de _ticker_list(). Bug regresion."
    )


def test_no_csv_legacy_en_cache_hit():
    """Ningun loader debe declarar cache_path apuntando al CSV legacy."""
    for rel, csv_name in [
        ("src/data_loader.py", "market_data.csv"),
        ("src/stock_data_loader.py", "stock_prices.csv"),
    ]:
        src = (ROOT / rel).read_text(encoding="utf-8-sig", errors="replace")
        assert f"cache_path = 'data/{csv_name}'" not in src, (
            f"{rel} sigue declarando cache_path para {csv_name}. "
            f"Bug: el cache-hit puede resucitar el CSV legacy."
        )
        assert "os.path.exists(cache_path)" not in src, (
            f"{rel} sigue consultando cache_path (CSV legacy)."
        )


def test_min_coverage_hardcoded_eliminado():
    """AST: min_coverage=0.90 eliminado de loaders (usan
    CACHE_MIN_COVERAGE) y de stocks_data (usa default de
    resolve_effective_date)."""
    for rel in ("src/data_loader.py", "src/stock_data_loader.py"):
        src = (ROOT / rel).read_text(encoding="utf-8-sig", errors="replace")
        assert "min_coverage=0.90" not in src, (
            f"{rel} sigue con min_coverage=0.90 hardcoded. Usar CACHE_MIN_COVERAGE."
        )
        assert "CACHE_MIN_COVERAGE" in src, (
            f"{rel} no referencia CACHE_MIN_COVERAGE."
        )
    sd_src = (ROOT / "src" / "pipeline" / "stocks_data.py").read_text(
        encoding="utf-8-sig", errors="replace"
    )
    assert "resolve_effective_date(df_stocks, _eligible, min_coverage=0.90)" not in sd_src, (
        "stocks_data.py: resolve_effective_date con min_coverage=0.90 "
        "redundante (el default de la funcion ya es 0.90)."
    )


def test_stock_data_loader_sigue_validando_tickers():
    """Regresion del check original de stock_data_loader.py."""
    src = (ROOT / "src" / "stock_data_loader.py").read_text(encoding="utf-8-sig", errors="replace")
    assert "_missing = set(all_tickers) - _cached_tickers" in src, (
        "stock_data_loader.py perdio el check de tickers faltantes."
    )


def test_default_resolve_effective_date_sigue_090():
    """El default de resolve_effective_date es 0.90 por contrato. No se
    toca; solo se elimina la llamada redundante."""
    src = (ROOT / "src" / "effective_date.py").read_text(encoding="utf-8-sig", errors="replace")
    assert "min_coverage: float = 0.90" in src, (
        "effective_date.py perdio el default 0.90 de resolve_effective_date. "
        "Es el contrato de la funcion, no debe cambiar."
    )
