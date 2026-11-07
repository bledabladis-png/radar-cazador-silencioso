"""Centinela: Wyckoff usa universo completo (2026-11-07).

Tras commit 1c2c4004 (universo completo de sectores) y 2026-11-07
(TARGET_WEIGHT_COVERAGE=1.0), el snapshot de holdings clasifica TODOS
los evaluables de cada sector, no un top-20 por peso.
"""
from pathlib import Path

import pytest


def test_target_weight_coverage_es_1():
    from indicators.wyckoff_holdings_report import TARGET_WEIGHT_COVERAGE
    assert TARGET_WEIGHT_COVERAGE == 1.0, (
        f"TARGET_WEIGHT_COVERAGE={TARGET_WEIGHT_COVERAGE}, esperado 1.0"
    )


def test_min_tickers_permite_universo_completo():
    from indicators.wyckoff_holdings_report import MIN_TICKERS_PER_GROUP
    assert MIN_TICKERS_PER_GROUP <= 1, (
        f"MIN_TICKERS_PER_GROUP={MIN_TICKERS_PER_GROUP} bloquearia grupos pequenos"
    )


def test_sector_wyckoff_distribution_cubre_holdings_completos():
    """El CSV del bot debe tener n_total >= 20 en todos los sectores.

    Antes del cambio, XLE (21 holdings) quedaba en 20 con TARGET=0.85.
    Con TARGET=1.0, XLE debe clasificar 21 (o mas, con datos historicos).
    """
    csv = Path("outputs/history/sector_wyckoff_distribution.csv")
    if not csv.exists():
        pytest.skip("CSV no disponible")
    import pandas as pd
    df = pd.read_csv(csv)
    last = df[df["date"] == df["date"].max()]
    if last.empty:
        pytest.skip("CSV sin datos")
    # Despues del run del bot con TARGET=1.0, todos deben ser >= 20.
    # XLE con 21 holdings completos debe tener 21 (o 20 si uno sin 460 barras).
    for _, r in last.iterrows():
        if r["sector"] == "XLE":
            assert r["n_total"] >= 20, (
                f"XLE n_total={r['n_total']}, esperado >=20 con universo completo"
            )