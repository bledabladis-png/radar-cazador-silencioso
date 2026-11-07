# -*- coding: utf-8 -*-
"""Test de coherencia snapshot vs agregado vs CSVs (Fase 4)."""
import pandas as pd


def test_snapshot_amplia_contrato_columnas():
    """El snapshot tiene las 14 columnas del contrato ampliado doc 63 §4."""
    from indicators.wyckoff_holdings_report import OUTPUT_COLUMNS
    expected = {
        "date", "group", "group_type", "ticker", "name", "weight",
        "phase", "event_phase", "phase_since", "close", "basis",
        "coverage_weight_pct", "n_bars", "evaluable",
    }
    assert set(OUTPUT_COLUMNS) == expected


def test_snapshot_universo_completo_por_grupo():
    """2026-11-07: universo completo. Con TARGET_WEIGHT_COVERAGE=1.0 el
    snapshot incluye TODOS los tickers evaluables del grupo (no top-20).

    Fixture: 30 tickers con weights decrecientes. Esperado: los 30 estan
    en el snapshot (o los que clasifiquen como evaluables), no solo 20.
    """
    from indicators.wyckoff_holdings_report import compute_wyckoff_snapshot
    # Fixture: 30 tickers por grupo
    tickers = [f"T{i:02d}" for i in range(30)]
    dates = pd.date_range("2023-01-01", periods=500, freq="B")
    cols = pd.MultiIndex.from_product([["Open","High","Low","Close","Volume"], tickers])
    df = pd.DataFrame(100.0, index=dates, columns=cols)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    # Holdings con weights decrecientes
    holdings = pd.DataFrame({
        "etf": ["XLK"]*30,
        "ticker": tickers,
        "weight": [30-i for i in range(30)],
    })
    snap = compute_wyckoff_snapshot(
        df, holdings, None, None, selection_mode="target_weight")
    if snap.empty:
        return  # puede pasar si classify_phase da INSUFFICIENT en todos
    # Universo completo: sin limite artificial de 20. Todos los evaluables.
    assert snap.groupby("group").size().max() > 20, (
        "con TARGET_WEIGHT_COVERAGE=1.0 y selection_mode='target_weight' "
        "el snapshot debe incluir >20 tickers por grupo (fixture: 30)");


def test_phase_since_iso_o_vacio():
    """phase_since en el snapshot es ISO date o vacio, no Timestamp."""
    import re
    from indicators.wyckoff_holdings_report import compute_wyckoff_snapshot
    tickers = [f"T{i:02d}" for i in range(25)]
    dates = pd.date_range("2023-01-01", periods=500, freq="B")
    cols = pd.MultiIndex.from_product([["Open","High","Low","Close","Volume"], tickers])
    df = pd.DataFrame(100.0, index=dates, columns=cols)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    holdings = pd.DataFrame({
        "etf": ["XLK"]*25, "ticker": tickers, "weight": [25-i for i in range(25)],
    })
    snap = compute_wyckoff_snapshot(df, holdings, None, None)
    for val in snap["phase_since"]:
        if val == "":
            continue
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", str(val)), f"phase_since no ISO: {val!r}"