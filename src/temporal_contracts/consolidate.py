"""Consolidacion de df_market con temporal_meta (FU-021-5 Fase 3).

Dictamen Q-P.3: temporal_meta es dict paralelo (autoridad). df.attrs
puede usarse como espejo auxiliar, nunca como fuente de verdad.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.temporal_contracts.base import (
    MarketDataBundle,
    TemporalResolution,
    STATUS_OK,
    STATUS_STALE,
)


def build_temporal_meta(
    resolutions: dict,
    reference_date: datetime,
    run_id: str = "",
) -> dict:
    """Construye temporal_meta desde las resoluciones de los 9 contratos.

    Estructura:
        {
            'by_contract': {name: {effective_date, expected_date,
                                   lag_days, coverage, status}},
            'global_last_date': date | None,
            'reference_date': datetime,
            'run_id': str,
        }

    global_last_date = max(effective_date) de contratos OK/STALE.
    """
    by_contract = {}
    ok_stale_dates = []
    for name, res in resolutions.items():
        if not isinstance(res, TemporalResolution):
            raise TypeError(
                f"resolutions[{name!r}] no es TemporalResolution: {type(res)}"
            )
        by_contract[name] = {
            "effective_date": res.effective_date,
            "expected_date": res.expected_date,
            "lag_days": res.lag_days,
            "coverage": res.coverage,
            "status": res.status,
        }
        if res.status in (STATUS_OK, STATUS_STALE) and res.effective_date is not None:
            ok_stale_dates.append(res.effective_date)

    global_last_date = max(ok_stale_dates) if ok_stale_dates else None
    return {
        "by_contract": by_contract,
        "global_last_date": global_last_date,
        "reference_date": reference_date,
        "run_id": run_id,
    }


def consolidate(
    df_market: pd.DataFrame,
    resolutions: dict,
    reference_date: datetime,
    run_id: str = "",
) -> MarketDataBundle:
    """Retorna MarketDataBundle con df_market + temporal_meta.

    No modifica df_market. No adjunta atributos (df.attrs es espejo
    opcional fuera de esta funcion).
    """
    meta = build_temporal_meta(resolutions, reference_date, run_id)
    return MarketDataBundle(df_market=df_market, temporal_meta=meta)