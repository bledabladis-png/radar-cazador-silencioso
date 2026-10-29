"""Detector experimental de REACCUMULATION (DICT3-REACCUMULATION v4).

Aislado del sistema productivo. NO modifica classify_wyckoff_phase, ni
el orquestador, ni ningun CSV productivo. Salida unica:
outputs/audit/reaccumulation_events_<cohort>.csv (gitignored).

R1 contexto UP: usa _trend_component(df, ticker) de indicators.wyckoff_v1
(funcion pura, causal). Nota: el contrato v4 cita detect_trend_context
que NO existe en el codigo; se sustituye por _trend_component con umbral
congelado > 0.05.

Restriccion estructural clave: t_min = max(H_R1+L_PREV, W_V+W) = 460.
Sin eso, R3 no puede evaluarse (p < W_V) y el detector devuelve 0
episodios por diseño, no por criterio.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd

from indicators.wyckoff_v1 import _trend_component


# ---------- parametros congelados (v4) ----------
W = 40
W_V = 40
L_PREV = 60
H_R1 = 400  # auditor: 200 MA lenta + 120 contexto + 80 margen
TREND_UP_THRESHOLD = 0.05
CONSOLIDATION_MAX_WIDTH = 0.15
PULLBACK_V_NORM = -0.3
BREAKOUT_V_NORM = 0.5
BREAKOUT_TOL_FRAC = 0.05

PASS = "PASS"
NOT_TRIGGERED = "NOT_TRIGGERED"
NOT_EVALUABLE = "NOT_EVALUABLE"


@dataclass
class Episode:
    ticker: str
    date_context: str
    date_start: str
    date_confirmed: str
    S_b: float
    R_b: float
    w_b: float
    tol_b: float
    pullback_session: Optional[str]
    v_norm_p_pullback: Optional[float]
    v_norm_b_breakout: Optional[float]
    trend_context: Optional[float]
    r1_res: str
    r2_res: str
    r3_res: str
    r4_res: str
    r5_res: str


def _v_norm_at(volumes: pd.Series, t_pos: int) -> Optional[float]:
    if t_pos < W_V:
        return None
    window = volumes.iloc[t_pos - W_V:t_pos]
    if len(window) != W_V:
        return None
    if window.isna().any() or (window <= 0).any():
        return None
    v_t = volumes.iloc[t_pos]
    if pd.isna(v_t) or v_t <= 0:
        return None
    mean = float(window.mean())
    sd = float(window.std(ddof=0))
    if sd == 0.0:
        return None
    return (float(v_t) - mean) / sd


def _r1_context_up(trend_series: pd.Series, context_pos: int):
    if context_pos < H_R1:
        return NOT_EVALUABLE, None
    if context_pos < 0 or context_pos >= len(trend_series):
        return NOT_EVALUABLE, None
    val = trend_series.iloc[context_pos]
    if pd.isna(val):
        return NOT_EVALUABLE, None
    tv = float(val)
    if tv >= TREND_UP_THRESHOLD:
        return PASS, tv
    return NOT_TRIGGERED, tv


def _r2_consolidation(close: pd.Series, b_pos: int):
    if b_pos < W:
        return NOT_EVALUABLE, 0.0, 0.0, 0.0, 0.0
    window = close.iloc[b_pos - W:b_pos]
    if len(window) != W or window.isna().any() or (window <= 0).any():
        return NOT_EVALUABLE, 0.0, 0.0, 0.0, 0.0
    S_b = float(window.min())
    R_b = float(window.max())
    w_b = R_b - S_b
    if S_b <= 0 or R_b <= S_b or w_b <= 0:
        return NOT_EVALUABLE, 0.0, 0.0, 0.0, 0.0
    tol_b = BREAKOUT_TOL_FRAC * w_b
    if w_b / R_b > CONSOLIDATION_MAX_WIDTH:
        return NOT_TRIGGERED, S_b, R_b, w_b, tol_b
    return PASS, S_b, R_b, w_b, tol_b


def _r3_pullback(close, volumes, b_pos, S_b, tol_b):
    if b_pos < W:
        return NOT_EVALUABLE, None, None
    found_nt = False
    for p_pos in range(b_pos - 1, b_pos - W - 1, -1):
        if p_pos < 0:
            break
        c_p = close.iloc[p_pos]
        if pd.isna(c_p) or c_p <= 0:
            continue
        if float(c_p) > S_b + tol_b:
            continue
        v = _v_norm_at(volumes, p_pos)
        if v is None:
            continue
        if v <= PULLBACK_V_NORM:
            return PASS, p_pos, v
        found_nt = True
    if found_nt:
        return NOT_TRIGGERED, None, None
    return NOT_EVALUABLE, None, None


def _r4_confirmation(close, b_pos, R_b, tol_b):
    if b_pos + 2 >= len(close):
        return NOT_EVALUABLE
    thr = R_b + tol_b
    for off in (1, 2):
        c = close.iloc[b_pos + off]
        if pd.isna(c):
            return NOT_EVALUABLE
        if float(c) <= thr:
            return NOT_TRIGGERED
    return PASS


def _r5_breakout_volume(volumes, b_pos):
    v = _v_norm_at(volumes, b_pos)
    if v is None:
        return NOT_EVALUABLE, None
    if v >= BREAKOUT_V_NORM:
        return PASS, v
    return NOT_TRIGGERED, v


def detect_reaccumulation(df_ticker: pd.DataFrame, ticker: str) -> list:
    if df_ticker is None or len(df_ticker) < W + 5:
        return []
    df = df_ticker.sort_index()
    close = df["Close"].astype(float)
    volumes = df["Volume"].astype(float)
    n = len(close)
    trend_series = _trend_component(df, ticker)

    t_min = max(H_R1 + L_PREV, W_V + W)  # 460 con v4
    t = t_min
    episodes = []
    while t <= n - 3:
        b = t
        r1, tv = _r1_context_up(trend_series, b - L_PREV)
        if r1 != PASS:
            t += 1; continue
        r2, S_b, R_b, w_b, tol_b = _r2_consolidation(close, b)
        if r2 != PASS:
            t += 1; continue
        r3, p_pos, vp = _r3_pullback(close, volumes, b, S_b, tol_b)
        if r3 != PASS:
            t += 1; continue
        c_b = close.iloc[b]
        if pd.isna(c_b) or float(c_b) <= R_b + tol_b:
            t += 1; continue
        r5, vb = _r5_breakout_volume(volumes, b)
        if r5 != PASS:
            t += 1; continue
        r4 = _r4_confirmation(close, b, R_b, tol_b)
        if r4 != PASS:
            t += 1; continue
        ep = Episode(
            ticker=ticker,
            date_context=str(df.index[b - L_PREV].date()),
            date_start=str(df.index[b].date()),
            date_confirmed=str(df.index[b + 2].date()),
            S_b=S_b, R_b=R_b, w_b=w_b, tol_b=tol_b,
            pullback_session=str(df.index[p_pos].date()) if p_pos is not None else None,
            v_norm_p_pullback=vp,
            v_norm_b_breakout=vb,
            trend_context=tv,
            r1_res=r1, r2_res=r2, r3_res=r3, r4_res=r4, r5_res=r5,
        )
        episodes.append(asdict(ep))
        t = b + 3
    return episodes


def write_csv(episodes: list, out_path: str) -> None:
    import pathlib
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "ticker", "date_context", "date_start", "date_confirmed",
        "S_b", "R_b", "w_b", "tol_b", "pullback_session",
        "v_norm_p_pullback", "v_norm_b_breakout", "trend_context",
        "r1_res", "r2_res", "r3_res", "r4_res", "r5_res",
    ]
    pd.DataFrame(episodes, columns=cols).to_csv(p, index=False)