# -*- coding: utf-8 -*-
# Contador de fases Wyckoff confirmadas con H20 completo.
#
# Soporte para WYCKOFF-5BX (n_confirmed_H20_complete >= 550 post-2026-10-02).
# Instrumentacion provisional: la semantica exacta de H20_complete se
# definira en la spec v3 de Wyckoff 5b.X. Interpretacion actual:
# - CONFIRMED = event_phase in {ACCUMULATION, DISTRIBUTION}.
#   FORMING y post-break (MARKUP, MARKDOWN) NO cuentan.
# - H20_complete = al menos min_sessions (default 20) sesiones bursatiles
#   transcurridas entre phase_since y reference_date. Aproximacion con
#   pd.bdate_range (ignora festivos NYSE; revisar si la spec v3 lo exige).
# - Una cuenta por cada (ticker, phase_since) distinto. Si un ticker entra
#   en CONFIRMED varias veces a lo largo de su historia, cada entrada
#   cuenta por separado.
#
# Referencia: WYCKOFF-5BX-INSTRUMENTACION en REGISTRO.yaml.
from __future__ import annotations

from datetime import date

import pandas as pd


CONFIRMED_PHASES = frozenset({"ACCUMULATION", "DISTRIBUTION"})
DEFAULT_MIN_SESSIONS = 20
THRESHOLD_5BX = 550


def _to_date(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        return None
    try:
        return pd.Timestamp(v).date()
    except (TypeError, ValueError):
        return None


def _business_days_between(start, end):
    if start is None or end is None or end < start:
        return 0
    rng = pd.bdate_range(start=start, end=end)
    return max(0, len(rng) - 1)


def count_confirmed_h20_complete(
    df_holdings,
    reference_date=None,
    min_sessions=DEFAULT_MIN_SESSIONS,
):
    result = {
        "n_confirmed_h20_complete": 0,
        "n_confirmed_total": 0,
        "min_sessions": int(min_sessions),
        "by_phase": {"ACCUMULATION": 0, "DISTRIBUTION": 0},
        "reference_date": None,
        "threshold_5bx": THRESHOLD_5BX,
        "threshold_reached": False,
    }
    if df_holdings is None or df_holdings.empty:
        return result
    required = {"ticker", "event_phase", "phase_since"}
    if not required.issubset(df_holdings.columns):
        return result

    ref = _to_date(reference_date)
    if ref is None:
        if "date" in df_holdings.columns:
            ref = _to_date(df_holdings["date"].max())
        if ref is None:
            ref = date.today()
    result["reference_date"] = ref.isoformat() if ref else None

    df = df_holdings[df_holdings["event_phase"].isin(CONFIRMED_PHASES)].copy()
    if df.empty:
        return result

    df = df.drop_duplicates(subset=["ticker", "phase_since"])
    result["n_confirmed_total"] = len(df)

    n_complete = 0
    for _, row in df.iterrows():
        ps = _to_date(row.get("phase_since"))
        if ps is None:
            continue
        if _business_days_between(ps, ref) >= min_sessions:
            n_complete += 1
            phase = row.get("event_phase")
            if phase in result["by_phase"]:
                result["by_phase"][phase] += 1

    result["n_confirmed_h20_complete"] = n_complete
    result["threshold_reached"] = n_complete >= THRESHOLD_5BX
    return result
