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


def _dedup_preserving_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _empty_result(requested_date, n_eligible, status="INSUFFICIENT_COVERAGE"):
    return {
        "date": None,
        "requested_date": requested_date,
        "lag_days": None,
        "n_eligible": n_eligible,
        "n_observed": 0,
        "coverage": 0.0,
        "status": status,
    }


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
        - Si el index de prices contiene timestamps duplicados, se agrupa
          por nivel 0 aplicando .last() antes de calcular cobertura.
          Garantiza que close_df.loc[effective_date, present] sea siempre
          una Series, no un DataFrame.
        - `requested_date` es None si y solo si prices es None o vacio.
          En el resto de paths siempre es un Timestamp.
    """
    eligible = _dedup_preserving_order(eligible_tickers)
    n_eligible = len(eligible)

    # Casos degenerados: sin datos -> requested_date=None.
    if prices is None or prices.empty or n_eligible == 0:
        return _empty_result(None, n_eligible)

    requested_date = prices.index[-1]

    # Normalizar a DataFrame con columnas planas (tickers).
    if isinstance(prices.columns, pd.MultiIndex):
        close_cols = [c for c in prices.columns if c[0] == 'Close']
        close_df = prices[close_cols].copy()
        close_df.columns = [c[1] for c in close_cols]
    else:
        close_df = prices

    # Index duplicado: agrupar por nivel 0 y tomar la ultima observacion
    # por fecha. Garantiza acceso escalar en close_df.loc[date, present].
    if close_df.index.has_duplicates:
        close_df = close_df.groupby(level=0, sort=True).last()

    present = [t for t in eligible if t in close_df.columns]

    if not present:
        return _empty_result(requested_date, n_eligible)

    coverage_series = close_df[present].notna().sum(axis=1) / n_eligible
    valid_rows = coverage_series[coverage_series >= min_coverage]

    if valid_rows.empty:
        return _empty_result(requested_date, n_eligible)

    effective_date = valid_rows.index[-1]
    row = close_df.loc[effective_date, present]
    # Con index agrupado y present deduplicado, row es Series.
    # Defensa adicional: si por alguna razon llegara un DataFrame,
    # colapsar a Series.
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    n_observed = int(row.notna().sum())
    coverage = n_observed / n_eligible

    try:
        lag_days = (
            pd.Timestamp(requested_date) - pd.Timestamp(effective_date)
        ).days
    except (TypeError, ValueError):
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
