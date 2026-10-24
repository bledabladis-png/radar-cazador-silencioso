"""Cascada de acumulacion Wyckoff: SC -> AR -> Range -> SOS -> ACCUMULATION.

Simetrica a wyckoff_distribution.py pero con direccion invertida:
- SC (Selling Climax): minimo con volumen alto + volatilidad alta + tendencia bajista previa
- AR (Automatic Rally): rebote desde SC que define la resistencia del rango
- Range: soporte = SC low, resistencia = AR high
- SOS (Sign of Strength): ruptura de resistencia del rango con volumen

NO reutiliza detectores de distribucion. Cada cascada es independiente para
evitar contaminacion entre ambas (requisito de diseno).
"""
from __future__ import annotations

import pandas as pd

from src.utils import get_col
from indicators.wyckoff_v1 import wyckoff_score
from indicators.wyckoff_event import (
    WyckoffEvent, Range, ClassifiedBreak,
    RangeType, Classification,
)


# Parametros PROVISIONALES. NO calibrados. Pre-registro pendiente.
SC_LOOKBACK = 60
SC_V_NORM_THRESHOLD = 0.5
SC_C_NORM_THRESHOLD = -0.3
SC_T_NORM_THRESHOLD = -0.5
MIN_OBS_SC = 80

AR_MIN_BOUNCE_PCT = 0.05
AR_MIN_PULLBACK_PCT = 0.03
AR_MAX_WINDOW = 30

RANGE_TOLERANCE_FRAC = 0.05

SOS_CONFIRM_CLOSES = 3
SOS_TIMEOUT_SESSIONS = 10

# Parametros ST (Secondary Test) - PROVISIONALES, pre-registro pendiente.
ST_MAX_WINDOW = 40
ST_PROXIMITY_PCT = 0.03
ST_MAX_RALLY_PCT = 0.05


def detect_sc(df, ticker, *, lookback=None, v_threshold=None,
              c_threshold=None, t_threshold=None, scan_window=1) -> WyckoffEvent:
    """Selling Climax: low[t] es el minimo del lookback con volumen extremo,
    volatilidad alta y tendencia bajista previa.

    Todas las features son observables al cierre de t.
    """
    lb = lookback if lookback is not None else SC_LOOKBACK
    vt = v_threshold if v_threshold is not None else SC_V_NORM_THRESHOLD
    ct = c_threshold if c_threshold is not None else SC_C_NORM_THRESHOLD
    tt = t_threshold if t_threshold is not None else SC_T_NORM_THRESHOLD

    if scan_window < 1:
        raise ValueError(f"scan_window({scan_window}) debe ser >= 1")

    try:
        low = get_col(df, ticker, "Low")
    except (KeyError, ValueError):
        return WyckoffEvent(
            ticker=ticker, event_type="SC",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "missing_low"},
        )

    low_clean = low.dropna()
    if len(low_clean) < max(lb, MIN_OBS_SC):
        return WyckoffEvent(
            ticker=ticker, event_type="SC",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "insufficient_data", "n": int(len(low_clean))},
        )

    try:
        _, _, _, t_norm, c_norm, v_norm, _ = wyckoff_score(df, ticker)
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return WyckoffEvent(
            ticker=ticker, event_type="SC",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": f"score_error: {exc}"},
        )

    idx = low.index
    n = len(idx)
    rmin = low_clean.dropna().rolling(lb, min_periods=lb).min().reindex(idx)
    t_al = t_norm.reindex(idx)
    c_al = c_norm.reindex(idx)
    v_al = v_norm.reindex(idx)

    if scan_window == 1:
        positions = [n - 1]
    else:
        positions = list(range(max(0, n - scan_window), n))[::-1]

    n_eval = 0
    n_nan = 0
    for pos in positions:
        bar_date = idx[pos]
        lo = low.iloc[pos]
        r = rmin.iloc[pos]
        t = t_al.iloc[pos]
        c = c_al.iloc[pos]
        v = v_al.iloc[pos]
        if any(pd.isna(x) for x in (lo, r, t, c, v)):
            n_nan += 1
            continue
        n_eval += 1
        if (float(lo) <= float(r) and float(v) >= vt
                and float(c) <= ct and float(t) <= tt):
            return WyckoffEvent(
                ticker=ticker, event_type="SC",
                event_date=bar_date, available_at=bar_date,
                status="CONFIRMED",
                evidence={
                    "low": float(lo), "roll_min": float(r),
                    "t_norm": float(t), "c_norm": float(c), "v_norm": float(v),
                    "lookback": int(lb), "scan_window": int(scan_window),
                    "bar_pos": int(pos), "bars_scanned": int(n_eval),
                    "nan_bars": int(n_nan),
                },
            )

    if n_eval == 0 and n_nan > 0:
        return WyckoffEvent(
            ticker=ticker, event_type="SC",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "nan_in_features", "nan_bars": int(n_nan)},
        )

    return WyckoffEvent(
        ticker=ticker, event_type="SC",
        event_date=None, available_at=None,
        status="NOT_OBSERVED",
        evidence={"reason": "no_sc_in_scan_window",
                  "scan_window": int(scan_window),
                  "lookback": int(lb),
                  "bars_scanned": int(n_eval),
                  "nan_bars": int(n_nan),
                  "v_threshold": float(vt),
                  "c_threshold": float(ct),
                  "t_threshold": float(tt)},
    )


def detect_ar_accumulation(df, ticker, sc_event, *, min_bounce_pct=None,
                           min_pullback_pct=None, max_ar_window=None) -> WyckoffEvent:
    """Automatic Rally tras SC confirmado.

    Requiere sc_event.status == CONFIRMED. Busca el High maximo en
    (sc_date, sc_date + max_ar_window] (excluye la barra SC).
    Si bounce_pct = (high - sc_low) / sc_low >= min_bounce_pct, el rally
    es candidato. Se confirma cuando un Close cae min_pullback_pct desde
    el high (incluyendo el propio high_date).
    """
    if sc_event.status != "CONFIRMED" or sc_event.event_date is None:
        return WyckoffEvent(
            ticker=ticker, event_type="AR_ACCUM",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "sc_not_confirmed", "sc_status": sc_event.status},
        )

    mb = min_bounce_pct if min_bounce_pct is not None else AR_MIN_BOUNCE_PCT
    mp = min_pullback_pct if min_pullback_pct is not None else AR_MIN_PULLBACK_PCT
    mw = max_ar_window if max_ar_window is not None else AR_MAX_WINDOW

    try:
        high = get_col(df, ticker, "High")
        close = get_col(df, ticker, "Close")
    except (KeyError, ValueError):
        return WyckoffEvent(
            ticker=ticker, event_type="AR_ACCUM",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "missing_high_close"},
        )

    sc_date = sc_event.event_date
    sc_low = sc_event.evidence.get("low")
    if sc_low is None or pd.isna(sc_low) or float(sc_low) <= 0:
        return WyckoffEvent(
            ticker=ticker, event_type="AR_ACCUM",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "sc_low_invalid"},
        )

    if sc_date not in df.index:
        return WyckoffEvent(
            ticker=ticker, event_type="AR_ACCUM",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "sc_date_not_in_df", "sc_date": str(sc_date)},
        )

    sc_pos = df.index.get_loc(sc_date)
    if sc_pos >= len(df) - 1:
        return WyckoffEvent(
            ticker=ticker, event_type="AR_ACCUM",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "no_post_sc_bars"},
        )

    end_pos = min(sc_pos + mw, len(df) - 1)
    window = high.iloc[sc_pos + 1:end_pos + 1].dropna()
    if window.empty:
        return WyckoffEvent(
            ticker=ticker, event_type="AR_ACCUM",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "no_ar_window_data"},
        )

    high_date = window.idxmax()
    high_val = float(window.loc[high_date])
    if not pd.notna(high_val) or high_val <= 0:
        return WyckoffEvent(
            ticker=ticker, event_type="AR_ACCUM",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "high_invalid", "high": str(high_val)},
        )

    bounce_pct = (high_val - float(sc_low)) / float(sc_low)

    evidence_base = {
        "sc_date": str(sc_date),
        "sc_low": float(sc_low),
        "high_date": str(high_date),
        "high": high_val,
        "bounce_pct": bounce_pct,
        "min_bounce_pct": float(mb),
        "min_pullback_pct": float(mp),
    }

    if bounce_pct < mb:
        return WyckoffEvent(
            ticker=ticker, event_type="AR_ACCUM",
            event_date=None, available_at=None,
            status="NOT_OBSERVED",
            evidence={**evidence_base, "reason": "bounce_below_threshold"},
        )

    high_pos = df.index.get_loc(high_date)
    pull_window = close.iloc[high_pos:].dropna()
    if pull_window.empty:
        return WyckoffEvent(
            ticker=ticker, event_type="AR_ACCUM",
            event_date=high_date, available_at=high_date,
            status="CANDIDATE",
            evidence={**evidence_base, "reason": "no_close_data_after_high"},
        )

    pullback = (high_val - pull_window) / high_val
    pull_mask = pullback >= mp
    if not bool(pull_mask.any()):
        return WyckoffEvent(
            ticker=ticker, event_type="AR_ACCUM",
            event_date=high_date, available_at=high_date,
            status="CANDIDATE",
            evidence={**evidence_base, "reason": "no_pullback_yet"},
        )

    pullback_date = pull_mask.idxmax()
    return WyckoffEvent(
        ticker=ticker, event_type="AR_ACCUM",
        event_date=high_date, available_at=pullback_date,
        status="CONFIRMED",
        evidence={**evidence_base,
                  "resistance": high_val,
                  "pullback_date": str(pullback_date),
                  "pullback_pct": float(pullback.loc[pullback_date])},
    )


def build_range_from_sc_ar(df, ticker, sc_event, ar_event, *,
                           range_id=None, range_type=None,
                           tolerance_frac=None, trend_context=None,
                           timeframe="1d") -> Range:
    """Construye un Range ACTIVO a partir de SC + AR confirmados.

    - support = sc_event.evidence["low"]
    - resistance = ar_event.evidence["resistance"]
    - candidate_start_date = sc_event.event_date
    - start_available_at = max(sc.available_at, ar.available_at)
    """
    if sc_event.status != "CONFIRMED":
        raise ValueError(f"sc_event no CONFIRMED: {sc_event.status}")
    if ar_event.status != "CONFIRMED":
        raise ValueError(f"ar_event no CONFIRMED: {ar_event.status}")
    if sc_event.event_date is None or ar_event.event_date is None:
        raise ValueError("SC o AR sin event_date")
    if sc_event.available_at is None or ar_event.available_at is None:
        raise ValueError("SC o AR sin available_at")

    support = sc_event.evidence.get("low")
    resistance = ar_event.evidence.get("resistance")
    if support is None or resistance is None:
        raise ValueError("SC sin low o AR sin resistance en evidence")
    support = float(support)
    resistance = float(resistance)
    if support >= resistance:
        raise ValueError(f"support({support}) >= resistance({resistance})")

    tf = tolerance_frac if tolerance_frac is not None else RANGE_TOLERANCE_FRAC
    width = resistance - support
    tol_abs = float(tf) * width

    rt: RangeType = range_type if range_type is not None else "UNCLASSIFIED"
    if rt not in ("ACCUMULATION_CANDIDATE", "UNCLASSIFIED"):
        raise ValueError(f"range_type invalido para acumulacion: {rt}")

    start_available = max(sc_event.available_at, ar_event.available_at)

    if range_id is None:
        range_id = (
            f"{ticker}_{timeframe}_ACC_{sc_event.event_date.date()}_"
            f"{ar_event.event_date.date()}"
        )

    evidence = {
        "sc_event_date": str(sc_event.event_date),
        "sc_available_at": str(sc_event.available_at),
        "sc_low": support,
        "ar_event_date": str(ar_event.event_date),
        "ar_available_at": str(ar_event.available_at),
        "ar_resistance": resistance,
        "width": width,
    }
    tc = dict(trend_context) if trend_context else {}

    return Range(
        range_id=range_id,
        ticker=ticker,
        timeframe=timeframe,
        formation_path="BC_AR",
        range_type=rt,
        candidate_start_date=sc_event.event_date,
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


def detect_sos_in_range(df, ticker, range_event, *, confirm_closes=None,
                        timeout_sessions=None, as_of=None) -> WyckoffEvent:
    """SOS anclado a un rango activo (breakout al alza).

    - Candidato: primer cierre > resistance + tol_abs en (range_active_at, as_of].
    - Confirmacion: confirm_closes cierres > resistance + tol_abs en
      [candidate_date, candidate_date + timeout_sessions].
    - Rechazo: algun cierre < resistance - tol_abs antes de confirmar.
    - Expiracion: pasan timeout_sessions sin confirmar ni rechazar.
    """
    if range_event.state != "RANGE_ACTIVE":
        return WyckoffEvent(
            ticker=ticker, event_type="SOS",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            range_id=range_event.range_id,
            evidence={"reason": "range_not_active", "range_state": range_event.state},
        )
    if range_event.range_active_at is None:
        return WyckoffEvent(
            ticker=ticker, event_type="SOS",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            range_id=range_event.range_id,
            evidence={"reason": "range_active_at_is_none"},
        )

    mc = confirm_closes if confirm_closes is not None else SOS_CONFIRM_CLOSES
    tw = timeout_sessions if timeout_sessions is not None else SOS_TIMEOUT_SESSIONS
    if mc < 1 or tw < 1:
        raise ValueError(f"confirm_closes({mc}) o timeout({tw}) invalidos")

    try:
        close = get_col(df, ticker, "Close")
    except (KeyError, ValueError):
        return WyckoffEvent(
            ticker=ticker, event_type="SOS",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            range_id=range_event.range_id,
            evidence={"reason": "missing_close"},
        )

    if as_of is None:
        as_of = df.index[-1]
    if as_of not in df.index:
        return WyckoffEvent(
            ticker=ticker, event_type="SOS",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            range_id=range_event.range_id,
            evidence={"reason": "as_of_not_in_index"},
        )
    if range_event.range_active_at not in df.index:
        return WyckoffEvent(
            ticker=ticker, event_type="SOS",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            range_id=range_event.range_id,
            evidence={"reason": "range_active_at_not_in_index"},
        )
    if as_of <= range_event.range_active_at:
        return WyckoffEvent(
            ticker=ticker, event_type="SOS",
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
            ticker=ticker, event_type="SOS",
            event_date=None, available_at=None,
            status="NOT_OBSERVED",
            range_id=range_event.range_id,
            evidence={"reason": "no_close_data_in_window"},
        )

    resistance = float(range_event.resistance)
    tol = float(range_event.tol_abs)
    lower_break = resistance + tol
    lower_recover = resistance - tol

    candidates = window[window > lower_break]
    if candidates.empty:
        return WyckoffEvent(
            ticker=ticker, event_type="SOS",
            event_date=None, available_at=None,
            status="NOT_OBSERVED",
            range_id=range_event.range_id,
            evidence={"reason": "no_close_above_resistance_plus_tol",
                      "resistance": resistance, "tol_abs": tol,
                      "lower_break": lower_break,
                      "lower_recover": lower_recover},
        )

    candidate_date = candidates.index[0]
    cand_pos = df.index.get_loc(candidate_date)
    c_close = float(close.loc[candidate_date])

    evidence_base = {
        "resistance": resistance,
        "tol_abs": tol,
        "lower_break": lower_break,
        "lower_recover": lower_recover,
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
        if c > lower_break:
            count += 1
            if count >= mc:
                return WyckoffEvent(
                    ticker=ticker, event_type="SOS",
                    event_date=candidate_date, available_at=candidate_date,
                    status="CONFIRMED",
                    range_id=range_event.range_id,
                    evidence={**evidence_base,
                              "resolution_event_date": str(bar_date),
                              "resolution_available_at": str(bar_date),
                              "closes_above": int(count),
                              "resolution_close": c},
                )
        else:
            if c < lower_recover:
                return WyckoffEvent(
                    ticker=ticker, event_type="SOS",
                    event_date=candidate_date, available_at=candidate_date,
                    status="REJECTED",
                    range_id=range_event.range_id,
                    evidence={**evidence_base,
                              "resolution_event_date": str(bar_date),
                              "resolution_available_at": str(bar_date),
                              "closes_above": int(count),
                              "resolution_close": c,
                              "reason": "close_below_resistance_minus_tol"},
                )

    timeout_pos = min(cand_pos + tw, len(df) - 1)
    timeout_date = df.index[timeout_pos]
    if timeout_pos >= len(df) - 1 and (cand_pos + tw) > len(df) - 1:
        return WyckoffEvent(
            ticker=ticker, event_type="SOS",
            event_date=candidate_date, available_at=candidate_date,
            status="CANDIDATE",
            range_id=range_event.range_id,
            evidence={**evidence_base, "closes_above": int(count),
                      "reason": "pending_as_of"},
        )

    return WyckoffEvent(
        ticker=ticker, event_type="SOS",
        event_date=candidate_date, available_at=candidate_date,
        status="EXPIRED",
        range_id=range_event.range_id,
        evidence={**evidence_base,
                  "resolution_event_date": str(timeout_date),
                  "resolution_available_at": str(timeout_date),
                  "closes_above": int(count),
                  "reason": "timeout_no_confirm_no_reject"},
    )


def _enrich_acc_evidence(evidence, sub, ticker, sos, rng):
    """Muta evidence in-place añadiendo mSOS y (si aplica) LPS.
    No cambia classification, solo evidencia pasiva."""
    from indicators.wyckoff_distribution import detect_msos, detect_lps
    try:
        msos = detect_msos(sub, ticker)
        evidence["msos"] = {"status": msos.status,
                            "event_date": str(msos.event_date) if msos.event_date else None}
    except Exception as exc:
        evidence["msos"] = {"status": "ERROR", "error": str(exc)}
    if sos is not None and sos.status == "CONFIRMED" and rng is not None:
        try:
            lps = detect_lps(sub, ticker, sos, rng)
            evidence["lps"] = {"status": lps.status,
                               "event_date": str(lps.event_date) if lps.event_date else None}
        except Exception as exc:
            evidence["lps"] = {"status": "ERROR", "error": str(exc)}
    return evidence


def classify_accumulation_at(df, ticker, as_of, *,
                             sc_kwargs=None, ar_kwargs=None,
                             range_kwargs=None, sos_kwargs=None,
                             trend_kwargs=None) -> ClassifiedBreak:
    """Cascada end-to-end de acumulacion: SC -> AR -> Range -> SOS -> clasificacion."""
    from indicators.wyckoff_distribution import detect_trend_context

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

    sck = dict(sc_kwargs or {})
    if "scan_window" not in sck:
        sck["scan_window"] = 500
    sc = detect_sc(sub, ticker, **sck)
    if sc.status != "CONFIRMED":
        return ClassifiedBreak(
            ticker=ticker, as_of=as_of, range_id=None,
            classification="NOT_EVALUABLE",
            structural_state="NOT_EVALUABLE",
            break_direction=None, range_type=None,
            trend_direction=None, sow_status=None,
            coherence_error=None,
            evidence={"reason": "sc_not_confirmed",
                      "sc_status": sc.status,
                      "sc_evidence": sc.evidence},
        )

    ar = detect_ar_accumulation(sub, ticker, sc, **(ar_kwargs or {}))
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

    trend = detect_trend_context(sub, ticker, sc.event_date, **(trend_kwargs or {}))

    rk = dict(range_kwargs or {})
    if "range_type" not in rk:
        if trend["direction"] == "DOWN":
            rk["range_type"] = "ACCUMULATION_CANDIDATE"
        else:
            rk["range_type"] = "UNCLASSIFIED"
    if "trend_context" not in rk:
        rk["trend_context"] = trend

    rng = build_range_from_sc_ar(sub, ticker, sc, ar, **rk)

    sos = detect_sos_in_range(sub, ticker, rng, as_of=as_of, **(sos_kwargs or {}))

    evidence = {
        "sc_date": str(sc.event_date),
        "ar_date": str(ar.event_date),
        "range_active_at": str(rng.range_active_at),
        "range_type": rng.range_type,
        "trend_direction": trend["direction"],
        "trend_evidence": trend["evidence"],
        "sos_status": sos.status,
        "sos_event_date": str(sos.event_date) if sos.event_date else None,
        "sos_evidence": sos.evidence,
    }
    # Evidencia pasiva: mSOS siempre, LPS si sos confirmado.
    _enrich_acc_evidence(evidence, sub, ticker, sos, rng)

    if sos.status == "CANDIDATE":
        return ClassifiedBreak(
            ticker=ticker, as_of=as_of, range_id=rng.range_id,
            classification="BREAKOUT_PENDING",
            structural_state="RANGE_ACTIVE",
            break_direction="UP",
            range_type=rng.range_type,
            trend_direction=trend["direction"],
            sow_status=sos.status,
            coherence_error=None,
            evidence=evidence,
        )

    if sos.status == "CONFIRMED":
        coherence_error = None
        if rng.range_type == "ACCUMULATION_CANDIDATE":
            if trend["direction"] != "DOWN":
                coherence_error = (
                    f"range_type=ACCUMULATION_CANDIDATE pero "
                    f"trend_direction={trend['direction']}"
                )
                classification: Classification = "BREAKOUT_WITHOUT_ACCUMULATION_CONTEXT"
            else:
                classification = "ACCUMULATION_CONFIRMED"
        else:
            classification = "BREAKOUT_WITHOUT_ACCUMULATION_CONTEXT"

        res_date_str = sos.evidence.get("resolution_event_date")
        try:
            res_date = pd.Timestamp(res_date_str) if res_date_str else None
        except (ValueError, TypeError):
            res_date = None
        return ClassifiedBreak(
            ticker=ticker, as_of=as_of, range_id=rng.range_id,
            classification=classification,
            structural_state="RANGE_BROKEN",
            break_direction="UP",
            range_type=rng.range_type,
            trend_direction=trend["direction"],
            sow_status=sos.status,
            coherence_error=coherence_error,
            evidence=evidence,
            range_active_at=rng.range_active_at,
            resolution_event_date=res_date,
            range_support=rng.support,
            range_resistance=rng.resistance,
            range_tol_abs=rng.tol_abs,
        )

    # sos.status in NOT_OBSERVED / REJECTED / EXPIRED / NOT_EVALUABLE
    # Si el rango tiene contexto de acumulacion (trend previo DOWN), la
    # acumulacion esta FORMANDOSE (aun sin SOS confirmado).
    if rng.range_type == "ACCUMULATION_CANDIDATE":
        cls = "ACCUMULATION_FORMING"
    else:
        cls = "RANGE_ACTIVE_NO_BREAK"
    return ClassifiedBreak(
        ticker=ticker, as_of=as_of, range_id=rng.range_id,
        classification=cls,
        structural_state="RANGE_ACTIVE",
        break_direction=None,
        range_type=rng.range_type,
        trend_direction=trend["direction"],
        sow_status=sos.status,
        coherence_error=None,
        evidence=evidence,
        range_active_at=rng.range_active_at,
        range_support=rng.support,
        range_resistance=rng.resistance,
        range_tol_abs=rng.tol_abs,
    )


def classify_accumulation(df, ticker, **kwargs) -> ClassifiedBreak:
    """Wrapper de classify_accumulation_at con as_of = ultima barra del df."""
    as_of = df.index[-1]
    return classify_accumulation_at(df, ticker, as_of, **kwargs)


def detect_st_acc(df, ticker, sc_event, *, max_window=None,
                  proximity_pct=None, max_rally_pct=None) -> WyckoffEvent:
    """Secondary Test tras SC confirmado (acumulacion).

    Simetrico a detect_st_dist. Busca un segundo acercamiento a la zona del
    SC (low del SC +- proximity). Confirma cuando:
    - low[t] <= sc_low * (1 + proximity_pct)  (retest de la zona por abajo)
    - close[t] > sc_low * (1 + max_rally_pct)  (rechazo, no cierre abajo)
    """
    if sc_event.status != "CONFIRMED" or sc_event.event_date is None:
        return WyckoffEvent(
            ticker=ticker, event_type="ST_ACC",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "sc_not_confirmed", "sc_status": sc_event.status},
        )

    mw = max_window if max_window is not None else ST_MAX_WINDOW
    px = proximity_pct if proximity_pct is not None else ST_PROXIMITY_PCT
    rl = max_rally_pct if max_rally_pct is not None else ST_MAX_RALLY_PCT

    try:
        low = get_col(df, ticker, "Low")
        close = get_col(df, ticker, "Close")
    except (KeyError, ValueError):
        return WyckoffEvent(
            ticker=ticker, event_type="ST_ACC",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "missing_low_close"},
        )

    sc_date = sc_event.event_date
    sc_low = sc_event.evidence.get("low")
    if sc_low is None or pd.isna(sc_low) or float(sc_low) <= 0:
        return WyckoffEvent(
            ticker=ticker, event_type="ST_ACC",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "sc_low_invalid"},
        )
    sc_low = float(sc_low)

    if sc_date not in df.index:
        return WyckoffEvent(
            ticker=ticker, event_type="ST_ACC",
            event_date=None, available_at=None,
            status="NOT_EVALUABLE",
            evidence={"reason": "sc_date_not_in_df"},
        )

    sc_pos = df.index.get_loc(sc_date)
    if sc_pos >= len(df) - 1:
        return WyckoffEvent(
            ticker=ticker, event_type="ST_ACC",
            event_date=None, available_at=None,
            status="NOT_OBSERVED",
            evidence={"reason": "no_post_sc_bars"},
        )

    end_pos = min(sc_pos + mw, len(df) - 1)
    proximity_band = sc_low * (1.0 + px)
    rejection_level = sc_low * (1.0 + rl)

    for pos in range(sc_pos + 1, end_pos + 1):
        bar_date = df.index[pos]
        lo = low.loc[bar_date]
        c = close.loc[bar_date]
        if pd.isna(lo) or pd.isna(c):
            continue
        lo = float(lo)
        c = float(c)
        if lo <= proximity_band and c > rejection_level:
            return WyckoffEvent(
                ticker=ticker, event_type="ST_ACC",
                event_date=bar_date, available_at=bar_date,
                status="CONFIRMED",
                evidence={
                    "sc_date": str(sc_date),
                    "sc_low": sc_low,
                    "st_low": lo,
                    "st_close": c,
                    "proximity_pct": float(px),
                    "max_rally_pct": float(rl),
                    "proximity_band": proximity_band,
                    "rejection_level": rejection_level,
                },
            )

    return WyckoffEvent(
        ticker=ticker, event_type="ST_ACC",
        event_date=None, available_at=None,
        status="NOT_OBSERVED",
        evidence={"reason": "no_st_in_window",
                  "sc_date": str(sc_date),
                  "sc_low": sc_low,
                  "max_window": int(mw),
                  "proximity_band": proximity_band,
                  "rejection_level": rejection_level},
    )

