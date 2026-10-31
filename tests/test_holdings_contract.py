# -*- coding: utf-8 -*-
"""Contrato HOLDINGS-NORMALIZACION-EN-ORIGEN (2026-10-31).

Cierra la clase O2/BRK-NORMALIZACION-RESIDUAL. Pre-registro en
docs/auditoria/wyckoff/79_holdings_normalizacion_origen_v1.md.

Invariantes verificados:
1. load_stocks_data emite ticker y ticker_norm (idempotente).
2. ensure_ticker_norm(df) idempotente: no toca dfs con ticker_norm,
   anade la columna si falta, tolera None y dfs sin ticker.
3. get_ticker_norm(row) resuelve Series y dict; prefiere
   ticker_norm, cae a normalize_yahoo_ticker.
4. Ningun consumidor de holdings_df llama normalize_yahoo_ticker
   directamente. Uso obligatorio de ensure_ticker_norm o
   get_ticker_norm.
"""
import ast
from pathlib import Path

import numpy as np

import pandas as pd
import pytest

from src.instrument_registry import (
    ensure_ticker_norm,
    get_ticker_norm,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

CONSUMIDORES = (
    "indicators/rs_internal.py",
    "indicators/sector_concentration.py",
    "indicators/stock_metrics.py",
    "indicators/wyckoff_holdings_report.py",
    "indicators/sector_breadth.py",
)


# --- 1. Contrato del productor (load_stocks_data) ---

def test_load_stocks_data_emite_ticker_y_ticker_norm(monkeypatch, tmp_path):
    """load_stocks_data anade ticker_norm al holdings_df."""
    csv = tmp_path / "holdings.csv"
    pd.DataFrame({
        "etf": ["XLF", "XLF"],
        "ticker": ["BRK.B", "AAPL"],
        "weight": [12.3, 5.6],
    }).to_csv(csv, index=False)

    import src.pipeline.stocks_data as sd

    # df_stocks sintetico con columnas MultiIndex (Close, ticker)
    cols = pd.MultiIndex.from_tuples(
        [("Close", "AAPL"), ("Close", "BRK-B")]
    )
    idx = pd.date_range("2026-10-29", periods=2)
    df_stocks = pd.DataFrame(
        np.random.randn(2, 2), columns=cols, index=idx
    )

    monkeypatch.setattr(sd, "ETF_HOLDINGS_PATH", str(csv))
    monkeypatch.setattr(sd, "download_stock_prices",
                        lambda **kw: df_stocks)
    monkeypatch.setattr(
        sd, "resolve_effective_date",
        lambda df, elig, **kw: {
            "status": "OK", "date": df.index[-1],
            "requested_date": df.index[-1], "lag_days": 0,
            "coverage": 1.0, "n_observed": len(elig),
            "n_eligible": len(elig),
        },
    )
    result = sd.load_stocks_data()
    hdf = result.get("holdings_df")
    assert hdf is not None, "holdings_df no se cargo"
    assert "ticker" in hdf.columns
    assert "ticker_norm" in hdf.columns
    assert list(hdf["ticker"]) == ["BRK.B", "AAPL"]
    assert list(hdf["ticker_norm"]) == ["BRK-B", "AAPL"]


# --- 2. ensure_ticker_norm (idempotencia y fallback) ---

def test_ensure_ticker_norm_none_y_sin_ticker():
    assert ensure_ticker_norm(None) is None
    df_sin = pd.DataFrame({"otra": [1, 2]})
    out = ensure_ticker_norm(df_sin)
    assert list(out.columns) == ["otra"]


def test_ensure_ticker_norm_anade_columna_si_falta():
    df = pd.DataFrame({"ticker": ["BRK.B", "AAPL"]})
    out = ensure_ticker_norm(df)
    assert "ticker_norm" in out.columns
    assert list(out["ticker_norm"]) == ["BRK-B", "AAPL"]
    # No muta el original
    assert "ticker_norm" not in df.columns


def test_ensure_ticker_norm_idempotente_si_ya_existe():
    df = pd.DataFrame({
        "ticker": ["BRK.B", "AAPL"],
        "ticker_norm": ["BRK-B", "AAPL"],
    })
    out = ensure_ticker_norm(df)
    # Mismo objeto, sin copia
    assert out is df


# --- 3. get_ticker_norm (Series y dict) ---

def test_get_ticker_norm_series_con_columna():
    row = pd.Series({"ticker": "BRK.B", "ticker_norm": "BRK-B"})
    assert get_ticker_norm(row) == "BRK-B"


def test_get_ticker_norm_series_sin_columna_fallback():
    row = pd.Series({"ticker": "BRK.B"})
    assert get_ticker_norm(row) == "BRK-B"


def test_get_ticker_norm_dict_con_y_sin_columna():
    assert get_ticker_norm({"ticker": "BRK.B", "ticker_norm": "BRK-B"}) == "BRK-B"
    assert get_ticker_norm({"ticker": "BRK.B"}) == "BRK-B"


def test_get_ticker_norm_tipo_invalido():
    with pytest.raises(TypeError):
        get_ticker_norm("BRK.B")


# --- 4. Centinela AST: consumidores no llaman normalize_yahoo_ticker ---

def _imports_normalize(tree):
    """Devuelve True si el modulo importa normalize_yahoo_ticker."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "normalize_yahoo_ticker":
                    return True
    return False


def _llama_normalize(tree):
    """Devuelve True si el modulo invoca normalize_yahoo_ticker(...)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "normalize_yahoo_ticker":
                return True
            if isinstance(f, ast.Attribute) and f.attr == "normalize_yahoo_ticker":
                return True
    return False


def test_consumidores_no_usan_normalize_yahoo_ticker_directamente():
    """Los consumidores de holdings_df deben usar ensure/get_ticker_norm.

    Si un consumidor nuevo aparece con normalize_yahoo_ticker(...),
    este test lo detecta antes de que llegue a produccion. Fix:
    usar ensure_ticker_norm(df) al principio del consumidor, o
    get_ticker_norm(row) si iterando filas.
    """
    violaciones = []
    for rel in CONSUMIDORES:
        p = REPO_ROOT / rel
        assert p.exists(), f"{rel} no existe (¿renombrado?)"
        tree = ast.parse(p.read_text(encoding="utf-8-sig"))
        if _llama_normalize(tree):
            violaciones.append(f"{rel}: invoca normalize_yahoo_ticker(...)")
        # El import esta permitido solo si no se invoca; en la practica
        # ambos van juntos. Avisamos si aparece el import sin uso.
    assert not violaciones, (
        "Consumidores que llaman normalize_yahoo_ticker directamente. "
        "Contrato HOLDINGS-NORMALIZACION-EN-ORIGEN: usar "
        "ensure_ticker_norm(df) o get_ticker_norm(row). "
        "Violaciones:\n" + "\n".join(violaciones)
    )


def test_consumidores_importan_ensure_o_get():
    """Cada consumidor importa ensure_ticker_norm o get_ticker_norm."""
    faltan = []
    for rel in CONSUMIDORES:
        p = REPO_ROOT / rel
        text = p.read_text(encoding="utf-8")
        if ("ensure_ticker_norm" not in text
                and "get_ticker_norm" not in text):
            faltan.append(rel)
    assert not faltan, (
        "Consumidores sin ensure_ticker_norm ni get_ticker_norm. "
        "Contrato HOLDINGS-NORMALIZACION-EN-ORIGEN. Faltan:\n"
        + "\n".join(faltan)
    )
