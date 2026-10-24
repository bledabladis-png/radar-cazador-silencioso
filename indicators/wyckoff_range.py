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

# Parametros AR - PROVISIONALES, pre-registro pendiente.
AR_MIN_DROP_PCT = 0.05
AR_MIN_BOUNCE_PCT = 0.03
AR_MAX_WINDOW = 30


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


def detect_ar(df, ticker, bc_event, *, min_drop_pct=None, min_bounce_pct=None,
              max_ar_window=None) -> WyckoffEvent:
    """Automatic Reaction tras BC confirmado.

    Requiere bc_event.status == CONFIRMED. Busca el Low minimo en
    (bc_date, bc_date + max_ar_window] (excluye la propia barra BC).
    Si drop_pct = (bc_high - low) / bc_high >= min_drop_pct, la reaccion
    es candidata. Se confirma cuando un Close (incluyendo el propio low_date)
    alcanza low * (1 + min_bounce_pct).

    - CONFIRMED: event_date = low_date, available_at = bounce_date,
      support = low.
    - CANDIDATE: drop confirmado, bounce pendiente.
      event_date = available_at = low_date.
    - NOT_OBSERVED: drop < min_drop_pct, o no hay low posterior al BC.
    - NOT_EVALUABLE: bc no CONFIRMED, o datos ausentes.
    """
    if bc_event.status != "CONFIRMED" or bc_event.event_date is None:
        return WyckoffEvent(
            ticker=ticker, event_type="AR",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "bc_not_confirmed",
                      "bc_status": bc_event.status},
        )

    md = min_drop_pct if min_drop_pct is not None else AR_MIN_DROP_PCT
    mb = min_bounce_pct if min_bounce_pct is not None else AR_MIN_BOUNCE_PCT
    mw = max_ar_window if max_ar_window is not None else AR_MAX_WINDOW

    try:
        low = get_col(df, ticker, "Low")
        close = get_col(df, ticker, "Close")
    except (KeyError, ValueError):
        return WyckoffEvent(
            ticker=ticker, event_type="AR",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "missing_ohlc"},
        )

    bc_date = bc_event.event_date
    bc_high = bc_event.evidence.get("high")
    if bc_high is None or pd.isna(bc_high) or float(bc_high) <= 0:
        return WyckoffEvent(
            ticker=ticker, event_type="AR",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "bc_high_invalid"},
        )

    if bc_date not in df.index:
        return WyckoffEvent(
            ticker=ticker, event_type="AR",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "bc_date_not_in_df",
                      "bc_date": str(bc_date)},
        )

    bc_pos = df.index.get_loc(bc_date)
    if bc_pos >= len(df) - 1:
        return WyckoffEvent(
            ticker=ticker, event_type="AR",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "no_post_bc_bars"},
        )

    end_pos = min(bc_pos + mw, len(df) - 1)
    window = low.iloc[bc_pos + 1:end_pos + 1].dropna()
    if window.empty:
        return WyckoffEvent(
            ticker=ticker, event_type="AR",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "no_ar_window_data"},
        )

    low_date = window.idxmin()
    low_val = float(window.loc[low_date])
    if not pd.notna(low_val) or low_val <= 0:
        return WyckoffEvent(
            ticker=ticker, event_type="AR",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "low_invalid", "low": str(low_val)},
        )

    drop_pct = (float(bc_high) - low_val) / float(bc_high)

    evidence_base = {
        "bc_date": str(bc_date),
        "bc_high": float(bc_high),
        "low_date": str(low_date),
        "low": low_val,
        "drop_pct": drop_pct,
        "min_drop_pct": float(md),
        "min_bounce_pct": float(mb),
    }

    if drop_pct < md:
        return WyckoffEvent(
            ticker=ticker, event_type="AR",
            event_date=None, available_at=None,
            status="NOT_OBSERVED",
            evidence={**evidence_base, "reason": "drop_below_threshold"},
        )

    low_pos = df.index.get_loc(low_date)
    bounce_window = close.iloc[low_pos:].dropna()
    if bounce_window.empty:
        return WyckoffEvent(
            ticker=ticker, event_type="AR",
            event_date=low_date, available_at=low_date,
            status="CANDIDATE",
            evidence={**evidence_base, "reason": "no_close_data_after_low"},
        )

    recovery = (bounce_window - low_val) / low_val
    bounce_mask = recovery >= mb
    if not bool(bounce_mask.any()):
        return WyckoffEvent(
            ticker=ticker, event_type="AR",
            event_date=low_date, available_at=low_date,
            status="CANDIDATE",
            evidence={**evidence_base, "reason": "no_bounce_yet"},
        )

    bounce_date = bounce_mask.idxmax()
    return WyckoffEvent(
        ticker=ticker, event_type="AR",
        event_date=low_date, available_at=bounce_date,
        status="CONFIRMED",
        evidence={**evidence_base,
                  "support": low_val,
                  "bounce_date": str(bounce_date),
                  "bounce_pct": float(recovery.loc[bounce_date])},
    )

