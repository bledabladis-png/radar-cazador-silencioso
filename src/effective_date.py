# -*- coding: utf-8 -*-
"""FU-020 (2026-09-15): resolucion de fecha efectiva con cobertura.

Determina la fecha mas reciente del DataFrame cuya cobertura de
observaciones validas sobre el universo elegible alcanza un minimo.

Regla arquitectonica (FU-020 / R2):
    Ninguna metrica agregada puede seleccionar la observacion temporal
    mediante la posicion fisica de la ultima fila. La fecha efectiva
    se resuelve explicitamente por resolve_effective_date().
"""
from collections.abc import Collection

import pandas as pd


def resolve_effective_date(
    prices: pd.DataFrame,
    eligible_tickers: Collection[str],
    min_coverage: float = 0.90,
) -> dict:
    """Resuelve la fecha mas reciente cuya cobertura alcanza el minimo.

    Args:
        prices: DataFrame con columnas = tickers. Index temporal.
        eligible_tickers: universo elegible para esta metrica.
        min_coverage: umbral [0, 1]. Default 0.90.

    Returns:
        {
            "date": Timestamp | None,
            "requested_date": Timestamp | None,
            "lag_days": int | None,
            "n_eligible": int,
            "n_observed": int,
            "coverage": float,
            "status": "OK" | "INSUFFICIENT_COVERAGE",
        }

    Reglas:
        - No lanza excepcion por falta de cobertura. Devuelve None / status
          INSUFFICIENT_COVERAGE.
        - Duplicados en eligible_tickers se deduplican preservando el orden.
        - Tickers elegibles sin columna en prices cuentan como NO observados.
        - coverage = observed / eligible.
    """
    # Deduplicar eligible preservando orden.
    seen = set()
    eligible = []
    for t in eligible_tickers:
        if t not in seen:
            seen.add(t)
            eligible.append(t)
    n_eligible = len(eligible)

    # Casos degenerados.
    if prices is None or prices.empty or n_eligible == 0:
        return {
            "date": None,
            "requested_date": None,
            "lag_days": None,
            "n_eligible": n_eligible,
            "n_observed": 0,
            "coverage": 0.0,
            "status": "INSUFFICIENT_COVERAGE",
        }

    requested_date = prices.index[-1]

    # Columnas presentes del universo elegible.
    present = [t for t in eligible if t in prices.columns]

    if not present:
        return {
            "date": None,
            "requested_date": requested_date,
            "lag_days": None,
            "n_eligible": n_eligible,
            "n_observed": 0,
            "coverage": 0.0,
            "status": "INSUFFICIENT_COVERAGE",
        }

    # Cobertura por fila: observados / eligible (no / present).
    coverage_series = prices[present].notna().sum(axis=1) / n_eligible
    valid_rows = coverage_series[coverage_series >= min_coverage]

    if valid_rows.empty:
        return {
            "date": None,
            "requested_date": requested_date,
            "lag_days": None,
            "n_eligible": n_eligible,
            "n_observed": 0,
            "coverage": 0.0,
            "status": "INSUFFICIENT_COVERAGE",
        }

    effective_date = valid_rows.index[-1]
    n_observed = int(prices.loc[effective_date, present].notna().sum())
    coverage = n_observed / n_eligible

    try:
        lag_days = (pd.Timestamp(requested_date) - pd.Timestamp(effective_date)).days
    except Exception:
        lag_days = None

    return {
        "date": effective_date,
        "requested_date": requested_date,
        "lag_days": lag_days,
        "n_eligible": n_eligible,
        "n_observed": n_observed,
        "coverage": coverage,
        "status": "OK",
    }
