# -*- coding: utf-8 -*-
"""
Distribucion de fases Wyckoff por sector v3.0 (Fase 4)

Consume el snapshot unico de holdings (`compute_wyckoff_snapshot` de
indicators/wyckoff_holdings_report.py). Universo: top-20 por peso,
n_bars >= 200 (contrato doc 63 §2.6).

Sustituye v2.0 (que iteraba TODOS los holdings con umbral 60 barras,
violando el contrato y divergiendo de los 6 CSVs).

Fases validas: ACCUMULATION, MARKUP, RANGE, DISTRIBUTION, MARKDOWN,
CONFLICT, mas los FORMING y CONFLICT. INSUFFICIENT_DATA se cuenta como
n_insufficient (no es una fase).

No alimenta motores, scores, pesos ni State Machine.
"""
import pandas as pd
import numpy as np

from src.utils import _observation_date_from_df
from config.tickers import MARKET_TICKERS

SECTORS = MARKET_TICKERS['sectors']
VALID_PHASES = [
    'ACCUMULATION', 'ACCUMULATION_FORMING', 'MARKUP',
    'RANGE',
    'DISTRIBUTION', 'DISTRIBUTION_FORMING', 'MARKDOWN',
    'CONFLICT',
]


def _get_snapshot(df_stocks, holdings_df, snapshot):
    """Devuelve el snapshot. Si no se pasa, lo calcula (compat tests)."""
    if snapshot is not None:
        return snapshot
    try:
        from indicators.wyckoff_holdings_report import compute_wyckoff_snapshot
    except ImportError as e:
        print(f"  [WARN] sector_wyckoff_distribution: import snapshot: {e}")
        return None
    return compute_wyckoff_snapshot(df_stocks, holdings_df, None, None)


def compute_sector_wyckoff_distribution(df_stocks, holdings_df, snapshot=None):
    """Agregado por sector desde el snapshot unico de holdings (Fase 4).

    Universo: top-20 por peso (contrato doc 63 §2.6), n_bars >= 200.
    Coherente con los 6 CSVs de wyckoff_holdings_report (mismo snapshot).
    """
    snap = _get_snapshot(df_stocks, holdings_df, snapshot)
    if snap is None or snap.empty:
        return pd.DataFrame()

    try:
        from indicators.wyckoff_holdings_report import USA_SECTOR_GROUP_TYPE
    except ImportError:
        USA_SECTOR_GROUP_TYPE = "sector_usa"

    snap = snap[
        (snap["group_type"] == USA_SECTOR_GROUP_TYPE)
        & (snap["group"].isin(SECTORS))
    ]
    if snap.empty:
        return pd.DataFrame()

    obs_date = _observation_date_from_df(df_stocks)
    rows = []
    for sector_etf in SECTORS:
        sub = snap[snap["group"] == sector_etf]
        if sub.empty:
            continue
        n_total = len(sub)
        n_valid = 0
        n_insufficient = 0
        phase_counts = {phase: 0 for phase in VALID_PHASES}
        for _, r in sub.iterrows():
            phase = r["phase"]
            if phase is None or phase == "INSUFFICIENT_DATA":
                n_insufficient += 1
                continue
            if phase in VALID_PHASES:
                phase_counts[phase] += 1
                n_valid += 1
            else:
                n_insufficient += 1
        coverage = (n_valid / n_total * 100) if n_total else np.nan
        row = {
            "date": obs_date,
            "sector": sector_etf,
            "n_total": n_total,
            "n_valid_wyckoff": n_valid,
            "n_insufficient_wyckoff": n_insufficient,
            "coverage_wyckoff": coverage,
        }
        if n_valid >= 5:
            for phase in VALID_PHASES:
                row[f"count_{phase.lower()}"] = phase_counts[phase]
                row[f"pct_{phase.lower()}"] = phase_counts[phase] / n_valid * 100
        else:
            for phase in VALID_PHASES:
                row[f"count_{phase.lower()}"] = phase_counts[phase]
                row[f"pct_{phase.lower()}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
