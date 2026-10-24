from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from src.utils import get_col
from indicators.wyckoff_v1 import wyckoff_score

EventStatus = Literal[
    "CONFIRMED", "CANDIDATE", "NOT_OBSERVED", "NOT_EVALUABLE", "INVALIDATED",
]

# Parametros PROVISIONALES. NO calibrados. Pre-registro pendiente.
# Se conservan aqui como defaults para permitir tests y desarrollo; el
# pre-registro debe fijarlos ANTES de cualquier evaluacion confirmatoria.
BC_LOOKBACK = 60
BC_V_NORM_THRESHOLD = 0.5
BC_C_NORM_THRESHOLD = -0.3
BC_T_NORM_THRESHOLD = 0.5
MIN_OBS_BC = 80


@dataclass(frozen=True)
class WyckoffEvent:
    ticker: str
    event_type: str
    event_date: pd.Timestamp | None
    available_at: pd.Timestamp | None
    status: EventStatus
    range_id: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


def detect_bc(df, ticker, *, lookback=None, v_threshold=None,
              c_threshold=None, t_threshold=None) -> WyckoffEvent:
    """Buying Climax: high[t] es el maximo del lookback con volumen
    extremo, volatilidad alta y tendencia alcista previa.

    Todas las features son observables al cierre de t:
    - high[t] == max(high[t-lb+1 : t])  (rolling max incluye t)
    - v_norm[t] >= threshold  (volumen elevado)
    - c_norm[t] <= threshold  (c_norm = -z(compression); bajo = volatil)
    - t_norm[t] >= threshold  (tendencia alcista)

    Devuelve un unico evento sobre el ultimo indice del df.
    No usa datos posteriores a t por construccion.
    """
    lb = lookback if lookback is not None else BC_LOOKBACK
    vt = v_threshold if v_threshold is not None else BC_V_NORM_THRESHOLD
    ct = c_threshold if c_threshold is not None else BC_C_NORM_THRESHOLD
    tt = t_threshold if t_threshold is not None else BC_T_NORM_THRESHOLD

    try:
        high = get_col(df, ticker, "High")
    except (KeyError, ValueError):
        return WyckoffEvent(
            ticker=ticker, event_type="BC",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "missing_high"},
        )

    high_clean = high.dropna()
    if len(high_clean) < max(lb, MIN_OBS_BC):
        return WyckoffEvent(
            ticker=ticker, event_type="BC",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "insufficient_data", "n": int(len(high_clean))},
        )

    try:
        _, struct_s, _, t_norm, c_norm, v_norm, _ = wyckoff_score(df, ticker)
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return WyckoffEvent(
            ticker=ticker, event_type="BC",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": f"score_error: {exc}"},
        )

    idx = high.index
    last = idx[-1]
    rmax = high_clean.rolling(lb, min_periods=lb).max().reindex(idx).loc[last]
    h_last = high.loc[last]
    t_last = t_norm.reindex(idx).loc[last]
    c_last = c_norm.reindex(idx).loc[last]
    v_last = v_norm.reindex(idx).loc[last]

    if any(pd.isna(x) for x in (h_last, rmax, t_last, c_last, v_last)):
        return WyckoffEvent(
            ticker=ticker, event_type="BC",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "nan_in_features"},
        )

    is_max = bool(float(h_last) >= float(rmax))
    is_vol = bool(float(v_last) >= vt)
    is_volat = bool(float(c_last) <= ct)
    is_uptrend = bool(float(t_last) >= tt)
    evidence = {
        "high": float(h_last),
        "roll_max": float(rmax),
        "t_norm": float(t_last),
        "c_norm": float(c_last),
        "v_norm": float(v_last),
        "lookback": int(lb),
        "is_max": is_max,
        "is_vol": is_vol,
        "is_volat": is_volat,
        "is_uptrend": is_uptrend,
    }

    if is_max and is_vol and is_volat and is_uptrend:
        return WyckoffEvent(
            ticker=ticker, event_type="BC",
            event_date=last, available_at=last,
            status="CONFIRMED",
            evidence=evidence,
        )

    return WyckoffEvent(
        ticker=ticker, event_type="BC",
        event_date=None, available_at=None,
        status="NOT_OBSERVED",
        evidence=evidence,
    )