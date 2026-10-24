from __future__ import annotations


import pandas as pd

from src.utils import get_col
from indicators.wyckoff_v1 import wyckoff_score
from indicators.wyckoff_event import (
    WyckoffEvent, Range, ClassifiedBreak,
    RangeType, Classification,
)


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

# Tolerancia del rango - PROVISIONAL, pre-registro pendiente.
RANGE_TOLERANCE_FRAC = 0.05

# Parametros SOW in range - PROVISIONALES, pre-registro pendiente.
SOW_CONFIRM_CLOSES = 3
SOW_TIMEOUT_SESSIONS = 10

# Parametros ST (Secondary Test) - PROVISIONALES, pre-registro pendiente.
ST_MAX_WINDOW = 40
ST_PROXIMITY_PCT = 0.03
ST_MAX_DRAWDOWN_PCT = 0.05

# Parametros UT/UTAD y Spring - PROVISIONALES, pre-registro pendiente.
FB_MAX_WINDOW = 15
FB_CONFIRM_SESSIONS = 2

# Parametros trend context - PROVISIONAL, pre-registro pendiente.
TREND_CONTEXT_WINDOW = 120
TREND_CONTEXT_UP_THRESHOLD = 0.2
TREND_CONTEXT_DOWN_THRESHOLD = -0.2


def detect_bc(df, ticker, *, lookback=None, v_threshold=None,
              c_threshold=None, t_threshold=None, scan_window=1) -> WyckoffEvent:
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

    if scan_window < 1:
        raise ValueError(f"scan_window({scan_window}) debe ser >= 1")

    idx = high.index
    n = len(idx)
    rmax = high_clean.dropna().rolling(lb, min_periods=lb).max().reindex(idx)
    t_al = t_norm.reindex(idx)
    c_al = c_norm.reindex(idx)
    v_al = v_norm.reindex(idx)

    if scan_window == 1:
        positions = [n - 1]
    else:
        positions = list(range(max(0, n - scan_window), n))[::-1]

    n_evaluated = 0
    n_nan = 0
    for pos in positions:
        bar_date = idx[pos]
        h = high.iloc[pos]
        r = rmax.iloc[pos]
        t = t_al.iloc[pos]
        c = c_al.iloc[pos]
        v = v_al.iloc[pos]
        if any(pd.isna(x) for x in (h, r, t, c, v)):
            n_nan += 1
            continue
        n_evaluated += 1
        if (float(h) >= float(r) and float(v) >= vt
                and float(c) <= ct and float(t) >= tt):
            return WyckoffEvent(
                ticker=ticker, event_type="BC",
                event_date=bar_date, available_at=bar_date,
                status="CONFIRMED",
                evidence={
                    "high": float(h), "roll_max": float(r),
                    "t_norm": float(t), "c_norm": float(c), "v_norm": float(v),
                    "lookback": int(lb), "scan_window": int(scan_window),
                    "bar_pos": int(pos), "bars_scanned": int(n_evaluated),
                    "nan_bars": int(n_nan),
                    "is_max": True, "is_vol": True,
                    "is_volat": True, "is_uptrend": True,
                },
            )

    if n_evaluated == 0 and n_nan > 0:
        return WyckoffEvent(
            ticker=ticker, event_type="BC",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "nan_in_features", "nan_bars": int(n_nan)},
        )

    return WyckoffEvent(
        ticker=ticker, event_type="BC",
        event_date=None, available_at=None,
        status="NOT_OBSERVED",
        evidence={"reason": "no_bc_in_scan_window",
                  "scan_window": int(scan_window),
                  "lookback": int(lb),
                  "bars_scanned": int(n_evaluated),
                  "nan_bars": int(n_nan),
                  "v_threshold": float(vt),
                  "c_threshold": float(ct),
                  "t_threshold": float(tt)},
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


def build_range_from_bc_ar(df, ticker, bc_event, ar_event, *,
                           range_id=None, range_type=None,
                           tolerance_frac=None, trend_context=None,
                           timeframe="1d") -> Range:
    """Construye un Range ACTIVO a partir de BC + AR confirmados (camino A).

    - resistance = bc_event.evidence["high"]
    - support = ar_event.evidence["support"]
    - candidate_start_date = bc_event.event_date
    - start_available_at = max(bc.available_at, ar.available_at)
    - range_active_at = start_available_at (niveles disponibles al confirmar AR)
    - tol_abs = tolerance_frac * (resistance - support)

    range_type se pasa del caller (evidencia disponible al abrir). Si no se
    pasa, UNCLASSIFIED. La funcion no infiere range_type.

    Raises:
        ValueError: BC o AR no CONFIRMED, o support >= resistance.
    """
    if bc_event.status != "CONFIRMED":
        raise ValueError(f"bc_event no CONFIRMED: {bc_event.status}")
    if ar_event.status != "CONFIRMED":
        raise ValueError(f"ar_event no CONFIRMED: {ar_event.status}")
    if bc_event.event_date is None or ar_event.event_date is None:
        raise ValueError("BC o AR sin event_date")
    if bc_event.available_at is None or ar_event.available_at is None:
        raise ValueError("BC o AR sin available_at")

    resistance = bc_event.evidence.get("high")
    support = ar_event.evidence.get("support")
    if resistance is None or support is None:
        raise ValueError("BC sin high o AR sin support en evidence")
    resistance = float(resistance)
    support = float(support)
    if support >= resistance:
        raise ValueError(
            f"support({support}) >= resistance({resistance})"
        )

    tf = tolerance_frac if tolerance_frac is not None else RANGE_TOLERANCE_FRAC
    width = resistance - support
    tol_abs = float(tf) * width

    rt: RangeType = range_type if range_type is not None else "UNCLASSIFIED"
    if rt not in ("DISTRIBUTION_CANDIDATE",
                  "REDISTRIBUTION_CANDIDATE",
                  "UNCLASSIFIED"):
        raise ValueError(f"range_type invalido: {rt}")

    start_available = max(bc_event.available_at, ar_event.available_at)

    if range_id is None:
        range_id = (
            f"{ticker}_{timeframe}_{bc_event.event_date.date()}_"
            f"{ar_event.event_date.date()}"
        )

    evidence = {
        "bc_event_date": str(bc_event.event_date),
        "bc_available_at": str(bc_event.available_at),
        "bc_high": resistance,
        "ar_event_date": str(ar_event.event_date),
        "ar_available_at": str(ar_event.available_at),
        "ar_support": support,
        "width": width,
    }
    tc = dict(trend_context) if trend_context else {}

    return Range(
        range_id=range_id,
        ticker=ticker,
        timeframe=timeframe,
        formation_path="BC_AR",
        range_type=rt,
        candidate_start_date=bc_event.event_date,
        start_available_at=start_available,
        range_active_at=start_available,
        support=support,
        resistance=resistance,
        tolerance_frac=float(tf),
        tol_abs=tol_abs,
        state="RANGE_ACTIVE",
        trend_context=tc,
        evidence=evidence,
    )


def detect_sow_in_range(df, ticker, range_event, *, confirm_closes=None,
                        timeout_sessions=None, as_of=None) -> WyckoffEvent:
    """SOW anclado a un rango activo.

    - Candidato: primer cierre < support - tol_abs en (range_active_at, as_of].
    - Confirmacion: confirm_closes cierres < support - tol_abs en
      [candidate_date, candidate_date + timeout_sessions].
    - Rechazo: algun cierre > support + tol_abs antes de confirmar.
    - Expiracion: pasan timeout_sessions sin confirmar ni rechazar.

    Devuelve un unico WyckoffEvent. event_date = candidate_date.
    available_at = candidate_date. resolution_event_date y
    resolution_available_at solo poblados si CONFIRMED/REJECTED/EXPIRED.
    """
    if range_event.state != "RANGE_ACTIVE":
        return WyckoffEvent(
            ticker=ticker, event_type="SOW",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            range_id=range_event.range_id,
            evidence={"reason": "range_not_active",
                      "range_state": range_event.state},
        )
    if range_event.range_active_at is None:
        return WyckoffEvent(
            ticker=ticker, event_type="SOW",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            range_id=range_event.range_id,
            evidence={"reason": "range_active_at_is_none"},
        )

    mc = confirm_closes if confirm_closes is not None else SOW_CONFIRM_CLOSES
    tw = timeout_sessions if timeout_sessions is not None else SOW_TIMEOUT_SESSIONS
    if mc < 1 or tw < 1:
        raise ValueError(f"confirm_closes({mc}) o timeout({tw}) invalidos")

    try:
        close = get_col(df, ticker, "Close")
    except (KeyError, ValueError):
        return WyckoffEvent(
            ticker=ticker, event_type="SOW",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            range_id=range_event.range_id,
            evidence={"reason": "missing_close"},
        )

    if as_of is None:
        as_of = df.index[-1]
    if as_of not in df.index:
        return WyckoffEvent(
            ticker=ticker, event_type="SOW",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            range_id=range_event.range_id,
            evidence={"reason": "as_of_not_in_index"},
        )
    if range_event.range_active_at not in df.index:
        return WyckoffEvent(
            ticker=ticker, event_type="SOW",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            range_id=range_event.range_id,
            evidence={"reason": "range_active_at_not_in_index"},
        )
    if as_of <= range_event.range_active_at:
        return WyckoffEvent(
            ticker=ticker, event_type="SOW",
            event_date=None, available_at=None,
            status="NOT_OBSERVED",
            range_id=range_event.range_id,
            evidence={"reason": "no_bars_after_range_active",
                      "as_of": str(as_of),
                      "range_active_at": str(range_event.range_active_at)},
        )

    start_pos = df.index.get_loc(range_event.range_active_at)
    end_pos = df.index.get_loc(as_of)
    window = close.iloc[start_pos + 1:end_pos + 1].dropna()
    if window.empty:
        return WyckoffEvent(
            ticker=ticker, event_type="SOW",
            event_date=None, available_at=None,
            status="NOT_OBSERVED",
            range_id=range_event.range_id,
            evidence={"reason": "no_close_data_in_window"},
        )

    support = float(range_event.support)
    tol = float(range_event.tol_abs)
    upper_break = support - tol
    upper_recover = support + tol

    candidates = window[window < upper_break]
    if candidates.empty:
        return WyckoffEvent(
            ticker=ticker, event_type="SOW",
            event_date=None, available_at=None,
            status="NOT_OBSERVED",
            range_id=range_event.range_id,
            evidence={"reason": "no_close_below_support_minus_tol",
                      "support": support, "tol_abs": tol,
                      "upper_break": upper_break,
                      "upper_recover": upper_recover},
        )

    candidate_date = candidates.index[0]
    cand_pos = df.index.get_loc(candidate_date)
    c_close = float(close.loc[candidate_date])

    evidence_base = {
        "support": support,
        "tol_abs": tol,
        "upper_break": upper_break,
        "upper_recover": upper_recover,
        "candidate_close": c_close,
        "confirm_closes": int(mc),
        "timeout_sessions": int(tw),
    }

    count = 0
    max_pos = min(cand_pos + tw, len(df) - 1)
    for pos in range(cand_pos, max_pos + 1):
        bar_date = df.index[pos]
        c = close.loc[bar_date]
        if pd.isna(c):
            continue
        c = float(c)
        if c < upper_break:
            count += 1
            if count >= mc:
                return WyckoffEvent(
                    ticker=ticker, event_type="SOW",
                    event_date=candidate_date, available_at=candidate_date,
                    status="CONFIRMED",
                    range_id=range_event.range_id,
                    evidence={**evidence_base,
                              "resolution_event_date": str(bar_date),
                              "resolution_available_at": str(bar_date),
                              "closes_under": int(count),
                              "resolution_close": c},
                )
        else:
            if c > upper_recover:
                return WyckoffEvent(
                    ticker=ticker, event_type="SOW",
                    event_date=candidate_date, available_at=candidate_date,
                    status="REJECTED",
                    range_id=range_event.range_id,
                    evidence={**evidence_base,
                              "resolution_event_date": str(bar_date),
                              "resolution_available_at": str(bar_date),
                              "closes_under": int(count),
                              "resolution_close": c,
                              "reason": "close_above_support_plus_tol"},
                )

    timeout_pos = min(cand_pos + tw, len(df) - 1)
    timeout_date = df.index[timeout_pos]
    if timeout_pos >= len(df) - 1 and (cand_pos + tw) > len(df) - 1:
        return WyckoffEvent(
            ticker=ticker, event_type="SOW",
            event_date=candidate_date, available_at=candidate_date,
            status="CANDIDATE",
            range_id=range_event.range_id,
            evidence={**evidence_base, "closes_under": int(count),
                      "reason": "pending_as_of"},
        )

    return WyckoffEvent(
        ticker=ticker, event_type="SOW",
        event_date=candidate_date, available_at=candidate_date,
        status="EXPIRED",
        range_id=range_event.range_id,
        evidence={**evidence_base,
                  "resolution_event_date": str(timeout_date),
                  "resolution_available_at": str(timeout_date),
                  "closes_under": int(count),
                  "reason": "timeout_no_confirm_no_reject"},
    )


def detect_trend_context(df, ticker, as_of, *, window=None,
                         up_threshold=None, down_threshold=None):
    """Detector minimo de tendencia previa a as_of.

    Direccion sobre t_norm promedio en [as_of-window, as_of-1].
    - UP si mean(t_norm) >= up_threshold
    - DOWN si mean(t_norm) <= down_threshold
    - NONE en otro caso

    available_at = as_of (se conoce al cierre).
    """
    w = window if window is not None else TREND_CONTEXT_WINDOW
    ut = up_threshold if up_threshold is not None else TREND_CONTEXT_UP_THRESHOLD
    dt = down_threshold if down_threshold is not None else TREND_CONTEXT_DOWN_THRESHOLD

    try:
        _, _, _, t_norm, _, _, _ = wyckoff_score(df, ticker)
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return {"direction": "NONE", "available_at": as_of,
                "evidence": {"reason": f"score_error: {exc}"}}

    if as_of not in df.index:
        return {"direction": "NONE", "available_at": as_of,
                "evidence": {"reason": "as_of_not_in_index"}}

    pos = df.index.get_loc(as_of)
    if pos < w:
        return {"direction": "NONE", "available_at": as_of,
                "evidence": {"reason": "insufficient_warmup", "pos": pos}}

    window_t = t_norm.iloc[pos - w:pos].dropna()
    if window_t.empty:
        return {"direction": "NONE", "available_at": as_of,
                "evidence": {"reason": "all_nan_in_window"}}

    mean_t = float(window_t.mean())
    if mean_t >= ut:
        direction = "UP"
    elif mean_t <= dt:
        direction = "DOWN"
    else:
        direction = "NONE"

    return {
        "direction": direction,
        "available_at": as_of,
        "evidence": {
            "window": int(w),
            "n_obs": int(len(window_t)),
            "mean_t_norm": mean_t,
            "up_threshold": float(ut),
            "down_threshold": float(dt),
        },
    }


def classify_distribution_at(df, ticker, as_of, *,
                             bc_kwargs=None, ar_kwargs=None,
                             range_kwargs=None, sow_kwargs=None,
                             trend_kwargs=None) -> ClassifiedBreak:
    """Cascada end-to-end: BC -> AR -> Range -> SOW -> clasificacion.

    Solo evalua hasta as_of. No usa datos posteriores.

    Estados posibles:
    - NOT_EVALUABLE: BC/AR no confirmados, o rango no activo
    - RANGE_ACTIVE_NO_BREAK: rango activo pero SOW no disparo
    - BREAK_PENDING: SOW CANDIDATE
    - DISTRIBUTION_CONFIRMED: SOW CONFIRMED + range_type DISTRIBUTION + trend UP
    - REDISTRIBUTION_CONFIRMED: analogo con REDISTRIBUTION + DOWN
    - BREAK_WITHOUT_DISTRIBUTION_CONTEXT: SOW CONFIRMED sin contexto coherente
    """
    sub = df.loc[:as_of]
    if sub.empty:
        return ClassifiedBreak(
            ticker=ticker, as_of=as_of, range_id=None,
            classification="NOT_EVALUABLE",
            structural_state="NOT_EVALUABLE",
            break_direction=None, range_type=None,
            trend_direction=None, sow_status=None,
            coherence_error=None,
            evidence={"reason": "empty_df_up_to_as_of"},
        )

    bck = dict(bc_kwargs or {})
    if "scan_window" not in bck:
        bck["scan_window"] = 500
    bc = detect_bc(sub, ticker, **bck)
    if bc.status != "CONFIRMED":
        return ClassifiedBreak(
            ticker=ticker, as_of=as_of, range_id=None,
            classification="NOT_EVALUABLE",
            structural_state="NOT_EVALUABLE",
            break_direction=None, range_type=None,
            trend_direction=None, sow_status=None,
            coherence_error=None,
            evidence={"reason": "bc_not_confirmed",
                      "bc_status": bc.status,
                      "bc_evidence": bc.evidence},
        )

    ar = detect_ar(sub, ticker, bc, **(ar_kwargs or {}))
    if ar.status != "CONFIRMED":
        return ClassifiedBreak(
            ticker=ticker, as_of=as_of, range_id=None,
            classification="NOT_EVALUABLE",
            structural_state="NOT_EVALUABLE",
            break_direction=None, range_type=None,
            trend_direction=None, sow_status=None,
            coherence_error=None,
            evidence={"reason": "ar_not_confirmed",
                      "ar_status": ar.status,
                      "ar_evidence": ar.evidence},
        )

    trend = detect_trend_context(sub, ticker, bc.event_date, **(trend_kwargs or {}))

    rk = dict(range_kwargs or {})
    if "range_type" not in rk:
        if trend["direction"] == "UP":
            rk["range_type"] = "DISTRIBUTION_CANDIDATE"
        elif trend["direction"] == "DOWN":
            rk["range_type"] = "REDISTRIBUTION_CANDIDATE"
        else:
            rk["range_type"] = "UNCLASSIFIED"
    if "trend_context" not in rk:
        rk["trend_context"] = trend

    rng = build_range_from_bc_ar(sub, ticker, bc, ar, **rk)

    sow = detect_sow_in_range(sub, ticker, rng, as_of=as_of, **(sow_kwargs or {}))

    evidence = {
        "bc_date": str(bc.event_date),
        "ar_date": str(ar.event_date),
        "range_active_at": str(rng.range_active_at),
        "range_type": rng.range_type,
        "trend_direction": trend["direction"],
        "trend_evidence": trend["evidence"],
        "sow_status": sow.status,
        "sow_event_date": str(sow.event_date) if sow.event_date else None,
        "sow_evidence": sow.evidence,
    }

    if sow.status == "CANDIDATE":
        return ClassifiedBreak(
            ticker=ticker, as_of=as_of, range_id=rng.range_id,
            classification="BREAK_PENDING",
            structural_state="RANGE_ACTIVE",
            break_direction="DOWN",
            range_type=rng.range_type,
            trend_direction=trend["direction"],
            sow_status=sow.status,
            coherence_error=None,
            evidence=evidence,
        )

    if sow.status == "CONFIRMED":
        # Verificar coherencia range_type <-> trend
        coherence_error = None
        if rng.range_type == "DISTRIBUTION_CANDIDATE":
            if trend["direction"] != "UP":
                coherence_error = (
                    f"range_type=DISTRIBUTION_CANDIDATE pero "
                    f"trend_direction={trend['direction']}"
                )
                classification: Classification = "BREAK_WITHOUT_DISTRIBUTION_CONTEXT"
            else:
                classification = "DISTRIBUTION_CONFIRMED"
        elif rng.range_type == "REDISTRIBUTION_CANDIDATE":
            if trend["direction"] != "DOWN":
                coherence_error = (
                    f"range_type=REDISTRIBUTION_CANDIDATE pero "
                    f"trend_direction={trend['direction']}"
                )
                classification = "BREAK_WITHOUT_DISTRIBUTION_CONTEXT"
            else:
                classification = "REDISTRIBUTION_CONFIRMED"
        else:
            classification = "BREAK_WITHOUT_DISTRIBUTION_CONTEXT"

        res_date_str = sow.evidence.get("resolution_event_date")
        try:
            res_date = pd.Timestamp(res_date_str) if res_date_str else None
        except (ValueError, TypeError):
            res_date = None
        return ClassifiedBreak(
            ticker=ticker, as_of=as_of, range_id=rng.range_id,
            classification=classification,
            structural_state="RANGE_BROKEN",
            break_direction="DOWN",
            range_type=rng.range_type,
            trend_direction=trend["direction"],
            sow_status=sow.status,
            coherence_error=coherence_error,
            evidence=evidence,
            range_active_at=rng.range_active_at,
            resolution_event_date=res_date,
            range_support=rng.support,
            range_resistance=rng.resistance,
            range_tol_abs=rng.tol_abs,
        )

    # sow.status in NOT_OBSERVED / REJECTED / EXPIRED / NOT_EVALUABLE
    # Si el rango tiene contexto de distribucion (trend previo UP), la
    # distribucion esta FORMANDOSE (aun sin SOW confirmado).
    if rng.range_type == "DISTRIBUTION_CANDIDATE":
        cls = "DISTRIBUTION_FORMING"
    else:
        cls = "RANGE_ACTIVE_NO_BREAK"
    return ClassifiedBreak(
        ticker=ticker, as_of=as_of, range_id=rng.range_id,
        classification=cls,
        structural_state="RANGE_ACTIVE",
        break_direction=None,
        range_type=rng.range_type,
        trend_direction=trend["direction"],
        sow_status=sow.status,
        coherence_error=None,
        evidence=evidence,
        range_active_at=rng.range_active_at,
        range_support=rng.support,
        range_resistance=rng.resistance,
        range_tol_abs=rng.tol_abs,
    )


def classify_distribution(df, ticker, **kwargs) -> ClassifiedBreak:
    """Wrapper de classify_distribution_at con as_of = ultima barra del df."""
    as_of = df.index[-1]
    return classify_distribution_at(df, ticker, as_of, **kwargs)


def walk_distribution(df, ticker, *, start=None, end=None,
                      progress=False, **kwargs):
    """Recorre bar a bar entre start y end, emitendo una fila por transicion.

    Emite solo cuando cambia classification, structural_state, range_id,
    range_type, trend_direction, sow_status o coherence_error respecto a
    la fila anterior. Primera fila siempre se emite.

    Coste: O(n_bars x coste_classify). Lento por diseno (~130s/ticker real).
    Aceptable para tests de estabilidad temporal, no para produccion diaria.

    Devuelve DataFrame con columnas:
      date, classification, structural_state, range_id, range_type,
      trend_direction, sow_status, coherence_error, is_first, is_transition
    """
    if start is None:
        start = df.index[0]
    if end is None:
        end = df.index[-1]
    if start not in df.index or end not in df.index:
        raise ValueError("start o end no en df.index")
    if end < start:
        raise ValueError("end < start")

    start_pos = df.index.get_loc(start)
    end_pos = df.index.get_loc(end)

    _STATE_FIELDS = (
        "classification", "structural_state", "range_id", "range_type",
        "trend_direction", "sow_status", "coherence_error",
    )

    rows = []
    prev_state = None

    for i, pos in enumerate(range(start_pos, end_pos + 1)):
        bar_date = df.index[pos]
        try:
            r = classify_distribution_at(df, ticker, bar_date, **kwargs)
        except Exception as exc:
            rows.append({
                "date": bar_date,
                "classification": f"__EXCEPTION__:{type(exc).__name__}",
                "structural_state": "NOT_EVALUABLE",
                "range_id": None, "range_type": None,
                "trend_direction": None, "sow_status": None,
                "coherence_error": str(exc),
                "is_first": False,
                "is_transition": True,
            })
            if progress and (i + 1) % 100 == 0:
                print(f"  walk {ticker}: {i+1} bars procesados")
            continue

        state = tuple(getattr(r, f) for f in _STATE_FIELDS)
        is_first = prev_state is None
        is_transition = state != prev_state

        if is_transition:
            rows.append({
                "date": bar_date,
                "classification": r.classification,
                "structural_state": r.structural_state,
                "range_id": r.range_id,
                "range_type": r.range_type,
                "trend_direction": r.trend_direction,
                "sow_status": r.sow_status,
                "coherence_error": r.coherence_error,
                "is_first": is_first,
                "is_transition": is_transition,
            })

        prev_state = state
        if progress and (i + 1) % 100 == 0:
            print(f"  walk {ticker}: {i+1} bars procesados")

    if not rows:
        return pd.DataFrame(columns=[
            "date", "classification", "structural_state", "range_id",
            "range_type", "trend_direction", "sow_status",
            "coherence_error", "is_first", "is_transition",
        ])
    return pd.DataFrame(rows)


def detect_st_dist(df, ticker, bc_event, *, max_window=None,
                   proximity_pct=None, max_drawdown_pct=None) -> WyckoffEvent:
    """Secondary Test tras BC confirmado (distribucion).

    Busca un segundo acercamiento a la zona del BC (high del BC +- proximity)
    en el rango (bc_date+1, bc_date+max_window]. Se confirma cuando:
    - high[t] >= bc_high * (1 - proximity_pct)  (retest de la zona)
    - close[t] < bc_high * (1 - max_drawdown_pct)  (rechazo, no cierre arriba)

    No exige volumen inferior (esa condicion no es universal segun Wyckoff).

    - CONFIRMED: event_date = fecha del retest, available_at = mismo (observable al cierre).
    - NOT_OBSERVED: ventana sin retest.
    - NOT_EVALUABLE: BC no confirmado o datos insuficientes.
    """
    if bc_event.status != "CONFIRMED" or bc_event.event_date is None:
        return WyckoffEvent(
            ticker=ticker, event_type="ST_DIST",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "bc_not_confirmed", "bc_status": bc_event.status},
        )

    mw = max_window if max_window is not None else ST_MAX_WINDOW
    px = proximity_pct if proximity_pct is not None else ST_PROXIMITY_PCT
    dd = max_drawdown_pct if max_drawdown_pct is not None else ST_MAX_DRAWDOWN_PCT

    try:
        high = get_col(df, ticker, "High")
        close = get_col(df, ticker, "Close")
    except (KeyError, ValueError):
        return WyckoffEvent(
            ticker=ticker, event_type="ST_DIST",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "missing_high_close"},
        )

    bc_date = bc_event.event_date
    bc_high = bc_event.evidence.get("high")
    if bc_high is None or pd.isna(bc_high) or float(bc_high) <= 0:
        return WyckoffEvent(
            ticker=ticker, event_type="ST_DIST",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "bc_high_invalid"},
        )
    bc_high = float(bc_high)

    if bc_date not in df.index:
        return WyckoffEvent(
            ticker=ticker, event_type="ST_DIST",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "bc_date_not_in_df"},
        )

    bc_pos = df.index.get_loc(bc_date)
    if bc_pos >= len(df) - 1:
        return WyckoffEvent(
            ticker=ticker, event_type="ST_DIST",
            event_date=None, available_at=None,
            status="NOT_OBSERVED",
            evidence={"reason": "no_post_bc_bars"},
        )

    end_pos = min(bc_pos + mw, len(df) - 1)
    proximity_band = bc_high * (1.0 - px)
    rejection_level = bc_high * (1.0 - dd)

    for pos in range(bc_pos + 1, end_pos + 1):
        bar_date = df.index[pos]
        h = high.loc[bar_date]
        c = close.loc[bar_date]
        if pd.isna(h) or pd.isna(c):
            continue
        h = float(h)
        c = float(c)
        if h >= proximity_band and c < rejection_level:
            return WyckoffEvent(
                ticker=ticker, event_type="ST_DIST",
                event_date=bar_date, available_at=bar_date,
                status="CONFIRMED",
                evidence={
                    "bc_date": str(bc_date),
                    "bc_high": bc_high,
                    "st_high": h,
                    "st_close": c,
                    "proximity_pct": float(px),
                    "max_drawdown_pct": float(dd),
                    "proximity_band": proximity_band,
                    "rejection_level": rejection_level,
                },
            )

    return WyckoffEvent(
        ticker=ticker, event_type="ST_DIST",
        event_date=None, available_at=None,
        status="NOT_OBSERVED",
        evidence={"reason": "no_st_in_window",
                  "bc_date": str(bc_date),
                  "bc_high": bc_high,
                  "max_window": int(mw),
                  "proximity_band": proximity_band,
                  "rejection_level": rejection_level},
    )


def detect_false_breakout_up(df, ticker, range_event, *, max_window=None,
                              confirm_sessions=None, as_of=None) -> WyckoffEvent:
    """Falsa ruptura alcista (UT o UTAD) desde un rango activo.

    Busca un cierre > resistance + tol_abs (ruptura falsa) seguido de
    confirm_sessions cierres < resistance (retorno al rango) en <= max_window.

    NO distingue UT de UTAD por si mismo. El consumidor decide cual es
    segun contexto (UT: antes de SOW; UTAD: cerca de SOW). Ambos son el
    mismo evento observable: ruptura arriba rechazada.

    - CONFIRMED: event_date = breakout_date, available_at = confirm_date.
    - NOT_OBSERVED: no hay ruptura arriba en la ventana.
    - NOT_EVALUABLE: rango no activo o datos insuficientes.
    """
    if range_event.state != "RANGE_ACTIVE":
        return WyckoffEvent(
            ticker=ticker, event_type="UT",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            range_id=range_event.range_id,
            evidence={"reason": "range_not_active", "range_state": range_event.state},
        )
    if range_event.range_active_at is None:
        return WyckoffEvent(
            ticker=ticker, event_type="UT",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            range_id=range_event.range_id,
            evidence={"reason": "range_active_at_is_none"},
        )

    mw = max_window if max_window is not None else FB_MAX_WINDOW
    cs = confirm_sessions if confirm_sessions is not None else FB_CONFIRM_SESSIONS

    try:
        close = get_col(df, ticker, "Close")
    except (KeyError, ValueError):
        return WyckoffEvent(
            ticker=ticker, event_type="UT",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            range_id=range_event.range_id,
            evidence={"reason": "missing_close"},
        )

    if as_of is None:
        as_of = df.index[-1]
    if as_of not in df.index or range_event.range_active_at not in df.index:
        return WyckoffEvent(
            ticker=ticker, event_type="UT",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            range_id=range_event.range_id,
            evidence={"reason": "dates_not_in_index"},
        )

    start_pos = df.index.get_loc(range_event.range_active_at)
    end_pos = df.index.get_loc(as_of)
    if end_pos <= start_pos:
        return WyckoffEvent(
            ticker=ticker, event_type="UT",
            event_date=None, available_at=None,
            status="NOT_OBSERVED",
            range_id=range_event.range_id,
            evidence={"reason": "no_bars_after_range_active"},
        )

    resistance = float(range_event.resistance)
    tol = float(range_event.tol_abs)
    breakout_level = resistance + tol

    # Buscar ruptura arriba
    breakout_pos = None
    for pos in range(start_pos + 1, end_pos + 1):
        bar_date = df.index[pos]
        c = close.loc[bar_date]
        if pd.isna(c):
            continue
        if float(c) > breakout_level:
            breakout_pos = pos
            break

    if breakout_pos is None:
        return WyckoffEvent(
            ticker=ticker, event_type="UT",
            event_date=None, available_at=None,
            status="NOT_OBSERVED",
            range_id=range_event.range_id,
            evidence={"reason": "no_breakout_above",
                      "breakout_level": breakout_level,
                      "resistance": resistance, "tol_abs": tol},
        )

    breakout_date = df.index[breakout_pos]
    max_pos = min(breakout_pos + mw, end_pos)

    # Buscar retorno: N cierres < resistance
    confirm_count = 0
    for pos in range(breakout_pos, max_pos + 1):
        bar_date = df.index[pos]
        c = close.loc[bar_date]
        if pd.isna(c):
            confirm_count = 0
            continue
        c = float(c)
        if c < resistance:
            confirm_count += 1
            if confirm_count >= cs:
                return WyckoffEvent(
                    ticker=ticker, event_type="UT",
                    event_date=breakout_date, available_at=bar_date,
                    status="CONFIRMED",
                    range_id=range_event.range_id,
                    evidence={
                        "resistance": resistance,
                        "tol_abs": tol,
                        "breakout_level": breakout_level,
                        "breakout_close": float(close.loc[breakout_date]),
                        "confirm_close": c,
                        "confirm_sessions": int(cs),
                        "max_window": int(mw),
                    },
                )
        else:
            confirm_count = 0

    return WyckoffEvent(
        ticker=ticker, event_type="UT",
        event_date=breakout_date, available_at=None,
        status="CANDIDATE",
        range_id=range_event.range_id,
        evidence={
            "resistance": resistance,
            "tol_abs": tol,
            "breakout_level": breakout_level,
            "breakout_close": float(close.loc[breakout_date]),
            "max_window": int(mw),
            "reason": "breakout_not_yet_rejected",
        },
    )


def detect_false_breakout_down(df, ticker, range_event, *, max_window=None,
                                confirm_sessions=None, as_of=None) -> WyckoffEvent:
    """Falsa ruptura bajista (Spring) desde un rango activo.

    Simetrico a detect_false_breakout_up. Cierre < support - tol_abs
    seguido de confirm_sessions cierres > support.
    """
    if range_event.state != "RANGE_ACTIVE":
        return WyckoffEvent(
            ticker=ticker, event_type="SPRING",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            range_id=range_event.range_id,
            evidence={"reason": "range_not_active", "range_state": range_event.state},
        )
    if range_event.range_active_at is None:
        return WyckoffEvent(
            ticker=ticker, event_type="SPRING",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            range_id=range_event.range_id,
            evidence={"reason": "range_active_at_is_none"},
        )

    mw = max_window if max_window is not None else FB_MAX_WINDOW
    cs = confirm_sessions if confirm_sessions is not None else FB_CONFIRM_SESSIONS

    try:
        close = get_col(df, ticker, "Close")
    except (KeyError, ValueError):
        return WyckoffEvent(
            ticker=ticker, event_type="SPRING",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            range_id=range_event.range_id,
            evidence={"reason": "missing_close"},
        )

    if as_of is None:
        as_of = df.index[-1]
    if as_of not in df.index or range_event.range_active_at not in df.index:
        return WyckoffEvent(
            ticker=ticker, event_type="SPRING",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            range_id=range_event.range_id,
            evidence={"reason": "dates_not_in_index"},
        )

    start_pos = df.index.get_loc(range_event.range_active_at)
    end_pos = df.index.get_loc(as_of)
    if end_pos <= start_pos:
        return WyckoffEvent(
            ticker=ticker, event_type="SPRING",
            event_date=None, available_at=None,
            status="NOT_OBSERVED",
            range_id=range_event.range_id,
            evidence={"reason": "no_bars_after_range_active"},
        )

    support = float(range_event.support)
    tol = float(range_event.tol_abs)
    breakout_level = support - tol

    breakout_pos = None
    for pos in range(start_pos + 1, end_pos + 1):
        bar_date = df.index[pos]
        c = close.loc[bar_date]
        if pd.isna(c):
            continue
        if float(c) < breakout_level:
            breakout_pos = pos
            break

    if breakout_pos is None:
        return WyckoffEvent(
            ticker=ticker, event_type="SPRING",
            event_date=None, available_at=None,
            status="NOT_OBSERVED",
            range_id=range_event.range_id,
            evidence={"reason": "no_breakout_below",
                      "breakout_level": breakout_level,
                      "support": support, "tol_abs": tol},
        )

    breakout_date = df.index[breakout_pos]
    max_pos = min(breakout_pos + mw, end_pos)

    confirm_count = 0
    for pos in range(breakout_pos, max_pos + 1):
        bar_date = df.index[pos]
        c = close.loc[bar_date]
        if pd.isna(c):
            confirm_count = 0
            continue
        c = float(c)
        if c > support:
            confirm_count += 1
            if confirm_count >= cs:
                return WyckoffEvent(
                    ticker=ticker, event_type="SPRING",
                    event_date=breakout_date, available_at=bar_date,
                    status="CONFIRMED",
                    range_id=range_event.range_id,
                    evidence={
                        "support": support,
                        "tol_abs": tol,
                        "breakout_level": breakout_level,
                        "breakout_close": float(close.loc[breakout_date]),
                        "confirm_close": c,
                        "confirm_sessions": int(cs),
                        "max_window": int(mw),
                    },
                )
        else:
            confirm_count = 0

    return WyckoffEvent(
        ticker=ticker, event_type="SPRING",
        event_date=breakout_date, available_at=None,
        status="CANDIDATE",
        range_id=range_event.range_id,
        evidence={
            "support": support,
            "tol_abs": tol,
            "breakout_level": breakout_level,
            "breakout_close": float(close.loc[breakout_date]),
            "max_window": int(mw),
            "reason": "breakout_not_yet_recovered",
        },
    )

