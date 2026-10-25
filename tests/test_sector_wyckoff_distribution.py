# -*- coding: utf-8 -*-
"""Tests de compute_sector_wyckoff_distribution (Fase 4).

Reescritos 2026-10-25: la funcion consume el snapshot unico de holdings
(`compute_wyckoff_snapshot` de wyckoff_holdings_report), no clasifica
por su cuenta. Los tests construyen el snapshot directamente y lo
pasan como argumento, para aislar la agregacion del detector.

El test de coherencia (test_coherence_snapshot_vs_csvs) es el que
habria cazado la violacion del contrato doc 63 §2.6 que existio entre
sesion 3 y Fase 4 (agregado con todos los holdings + 60 barras,
CSVs con top-20 + 200 barras).
"""
import pandas as pd
import pytest


def _make_snapshot(phases_by_ticker, group="XLK", group_type="sector_usa"):
    """Construye un snapshot sintetico con OUTPUT_COLUMNS del contrato.

    phases_by_ticker: dict {ticker: phase}.
    """
    from indicators.wyckoff_holdings_report import OUTPUT_COLUMNS
    rows = []
    for i, (ticker, phase) in enumerate(phases_by_ticker.items()):
        rows.append({
            "date": "2026-10-23",
            "group": group,
            "group_type": group_type,
            "ticker": ticker,
            "name": ticker,
            "weight": float(30 - i),
            "phase": phase,
            "event_phase": phase if phase in ("ACCUMULATION", "DISTRIBUTION", "MARKUP", "MARKDOWN") else "RANGE",
            "phase_since": "",
            "close": 100.0,
            "basis": "",
            "coverage_weight_pct": 80.0,
            "n_bars": 500,
            "evaluable": True,
        })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def test_coverage_y_pct_calculados_por_la_funcion():
    """6 tickers ACCUMULATION -> n_valid=6 >= 5 -> pct_accumulation=100."""
    from indicators import sector_wyckoff_distribution as swd
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    snap = _make_snapshot({t: "ACCUMULATION" for t in tickers})

    out = swd.compute_sector_wyckoff_distribution(
        df_stocks=None, holdings_df=None, snapshot=snap,
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert row["sector"] == "XLK"
    assert row["n_total"] == 6
    assert row["n_valid_wyckoff"] == 6
    assert row["n_insufficient_wyckoff"] == 0
    assert row["coverage_wyckoff"] == pytest.approx(100.0)
    assert row["pct_accumulation"] == pytest.approx(100.0)
    for ph in ["markup", "range", "distribution", "markdown"]:
        assert row[f"pct_{ph}"] == pytest.approx(0.0)


def test_pct_nan_cuando_n_valid_menor_que_5():
    """3 tickers MARKUP -> n_valid=3 < 5 -> pct_* NaN, count_markup=3."""
    from indicators import sector_wyckoff_distribution as swd
    tickers = ["AAPL", "MSFT", "GOOGL"]
    snap = _make_snapshot({t: "MARKUP" for t in tickers})

    out = swd.compute_sector_wyckoff_distribution(
        df_stocks=None, holdings_df=None, snapshot=snap,
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_valid_wyckoff"] == 3
    assert row["coverage_wyckoff"] == pytest.approx(100.0)
    for ph in ["accumulation", "markup", "range", "distribution", "markdown"]:
        assert pd.isna(row[f"pct_{ph}"])
    assert row["count_markup"] == 3


def test_ignora_group_types_no_sectoriales():
    """Solo cuenta group_type='sector_usa'. Indices EU/USA fuera."""
    from indicators import sector_wyckoff_distribution as swd
    snap = pd.concat([
        _make_snapshot({f"T{i}": "ACCUMULATION" for i in range(10)}, group="XLK", group_type="sector_usa"),
        _make_snapshot({f"S{i}": "ACCUMULATION" for i in range(5)}, group="SPY", group_type="index_usa"),
        _make_snapshot({f"E{i}": "ACCUMULATION" for i in range(5)}, group="FEZ", group_type="index_eu"),
    ], ignore_index=True)

    out = swd.compute_sector_wyckoff_distribution(
        df_stocks=None, holdings_df=None, snapshot=snap,
    )
    # Solo XLK debe aparecer
    assert set(out["sector"].unique()) == {"XLK"}
    assert out.iloc[0]["n_total"] == 10


def test_snapshot_vacio_devuelve_dataframe_vacio():
    """Snapshot vacio -> DataFrame vacio, sin excepcion."""
    from indicators import sector_wyckoff_distribution as swd
    from indicators.wyckoff_holdings_report import OUTPUT_COLUMNS
    empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
    out = swd.compute_sector_wyckoff_distribution(
        df_stocks=None, holdings_df=None, snapshot=empty,
    )
    assert out.empty


def test_coherence_snapshot_vs_csvs():
    """CRITICO (D4.8 saliente): agregado y CSVs derivan del mismo snapshot.

    Construye un snapshot con phases mixtas para 11 sectores, agrega
    por sector y proyecta a ACC/DIST. Verifica que el count_* del
    agregado coincide exactamente con la cuenta de los CSVs filtrados.

    Si este test pasa en CI, la divergencia detectada en sesion 4
    (XLC/XLK/XLP agregado != CSVs) no puede reproducirse.
    """
    from indicators import sector_wyckoff_distribution as swd
    from indicators.wyckoff_holdings_report import (
        OUTPUT_COLUMNS, _tables_from_snapshot,
    )
    from config.tickers import MARKET_TICKERS

    sectors = MARKET_TICKERS["sectors"]
    phases = ["ACCUMULATION", "ACCUMULATION_FORMING", "MARKUP",
              "RANGE", "DISTRIBUTION", "DISTRIBUTION_FORMING",
              "MARKDOWN", "CONFLICT"]
    rows = []
    for i, sector in enumerate(sectors):
        for j in range(20):
            rows.append({
                "date": "2026-10-23",
                "group": sector,
                "group_type": "sector_usa",
                "ticker": f"T{sector}_{j:02d}",
                "name": f"T{sector}_{j}",
                "weight": float(20 - j),
                "phase": phases[(i + j) % len(phases)],
                "event_phase": "RANGE",
                "phase_since": "",
                "close": 100.0,
                "basis": "",
                "coverage_weight_pct": 80.0,
                "n_bars": 500,
                "evaluable": True,
            })
    snap = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    agg = swd.compute_sector_wyckoff_distribution(
        df_stocks=None, holdings_df=None, snapshot=snap,
    )
    tables = _tables_from_snapshot(snap)

    # Reconstruir counts por sector desde acc+dist
    acc = tables["usa_sector_acc"]
    dist = tables["usa_sector_dist"]
    for sector in sectors:
        agg_row = agg[agg["sector"] == sector].iloc[0]
        for phase in phases:
            expected = int((snap[(snap["group"] == sector)]["phase"] == phase).sum())
            got = int(agg_row[f"count_{phase.lower()}"])
            assert got == expected, (
                f"{sector}.{phase}: agregado={got} snapshot={expected}"
            )
        # ACC y DIST filtrados del snapshot coinciden con las tablas
        snap_sec = snap[snap["group"] == sector]
        acc_count = len(acc[acc["group"] == sector]) if not acc.empty else 0
        dist_count = len(dist[dist["group"] == sector]) if not dist.empty else 0
        # acc_count = ACC + ACC_FORMING + MARKUP, pero solo de phase in {ACCUMULATION, ACCUMULATION_FORMING, MARKUP}
        acc_from_snap = len(snap_sec[snap_sec["phase"].isin(
            {"ACCUMULATION", "ACCUMULATION_FORMING", "MARKUP"}
        )])
        dist_from_snap = len(snap_sec[snap_sec["phase"].isin(
            {"DISTRIBUTION", "DISTRIBUTION_FORMING", "MARKDOWN"}
        )])
        assert acc_count == acc_from_snap, f"{sector} acc: {acc_count} vs {acc_from_snap}"
        assert dist_count == dist_from_snap, f"{sector} dist: {dist_count} vs {dist_from_snap}"
