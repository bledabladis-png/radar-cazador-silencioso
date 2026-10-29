"""T1-T7 del detector REACCUMULATION (DICT3-REACCUMULATION v4)."""
import ast
import pathlib

import numpy as np
import pandas as pd

from scripts.audit.reaccumulation_detector import (
    detect_reaccumulation, TREND_UP_THRESHOLD, H_R1, L_PREV,
    PASS, NOT_TRIGGERED, NOT_EVALUABLE,
    _r1_context_up, _r2_consolidation, _r3_pullback,
    _r4_confirmation, _r5_breakout_volume,
)
from indicators.wyckoff_v1 import _trend_component


def _df_from_closes(closes, volumes=None):
    n = len(closes)
    idx = pd.bdate_range("2018-01-01", periods=n)
    if volumes is None:
        volumes = [1_000_000.0] * n
    return pd.DataFrame({
        "Open": closes, "High": closes, "Low": closes,
        "Close": closes, "Volume": volumes,
    }, index=idx)


def _noisy_volume(n, mean=1_000_000.0, amp=150_000.0):
    """Volumen con varianza determinista para que sd != 0 en ventanas."""
    return mean + amp * np.sin(np.arange(n) * 0.7)


def _make_t7_series():
    """Serie que satisface R1-R5 en b=540."""
    n = 700
    close = np.zeros(n)
    volume = _noisy_volume(n)
    # 400 sesiones planas
    close[:400] = 100.0
    # 100 sesiones alcistas
    close[400:500] = np.linspace(100.0, 120.0, 100)
    # 40 sesiones de consolidacion [500, 540)
    cons = np.zeros(40)
    for i in range(40):
        cons[i] = 118.0 if i % 2 == 0 else 130.0
    close[500:540] = cons
    # pullback en 510: close=118 (S_b) con volumen muy bajo
    volume[510] = 200_000.0
    # b=540: rompe con volumen alto
    close[540] = 135.0
    volume[540] = 3_000_000.0
    close[541] = 140.0
    close[542] = 145.0
    close[543:] = 145.0
    return pd.DataFrame({
        "Open": close, "High": close, "Low": close,
        "Close": close, "Volume": volume,
    }, index=pd.bdate_range("2018-01-01", periods=n))


# ----------------- T1: R1 contexto UP -----------------

def test_t1_r1_sin_historial_es_not_evaluable():
    df = _df_from_closes(np.full(500, 100.0))
    trend = _trend_component(df, "TEST")
    status, _ = _r1_context_up(trend, H_R1 - 1)
    assert status == NOT_EVALUABLE


def test_t1_r1_contexto_plano_no_es_up():
    df = _df_from_closes(np.full(700, 100.0))
    trend = _trend_component(df, "TEST")
    status, val = _r1_context_up(trend, 600)
    assert status == NOT_TRIGGERED
    assert val is not None and val < TREND_UP_THRESHOLD


def test_t1_r1_tendencia_alcista_es_up():
    closes = np.concatenate([np.full(500, 100.0), np.linspace(100, 150, 200)])
    df = _df_from_closes(closes)
    trend = _trend_component(df, "TEST")
    status, val = _r1_context_up(trend, 650)
    assert status == PASS
    assert val >= TREND_UP_THRESHOLD


# ----------------- T2: R2 consolidacion -----------------

def test_t2_r2_width_amplio_es_not_triggered():
    closes = np.array([100.0, 200.0] * 40)
    df = _df_from_closes(closes)
    status, *_ = _r2_consolidation(df["Close"], 40)
    assert status == NOT_TRIGGERED


def test_t2_r2_width_ok_es_pass():
    closes = np.array([118.0, 130.0] * 40)
    df = _df_from_closes(closes)
    status, S_b, R_b, w_b, tol_b = _r2_consolidation(df["Close"], 40)
    assert status == PASS
    assert S_b == 118.0 and R_b == 130.0
    assert abs(w_b - 12.0) < 1e-9
    assert abs(tol_b - 0.6) < 1e-9


def test_t2_r2_nan_es_not_evaluable():
    closes = np.array([118.0] * 40)
    df = _df_from_closes(closes)
    df.loc[df.index[5], "Close"] = np.nan
    status, *_ = _r2_consolidation(df["Close"], 40)
    assert status == NOT_EVALUABLE


# ----------------- T3: R3 pullback -----------------

def test_t3_r3_pullback_valido():
    """b=100, ventana [60, 99], pullback en p=70 con volumen bajo."""
    n = 103
    close = np.full(n, 130.0)
    close[60:100] = [118.0 if i % 2 == 0 else 130.0 for i in range(40)]
    close[100] = 135.0
    close[101] = 140.0
    close[102] = 145.0
    volume = _noisy_volume(n)
    volume[70] = 100_000.0  # v_norm(70) muy bajo
    df = _df_from_closes(close, volume)
    status, p_pos, v = _r3_pullback(df["Close"], df["Volume"], 100, 118.0, 0.6)
    assert status == PASS, f"status={status} p_pos={p_pos} v={v}"
    assert p_pos is not None
    assert v <= -0.3


def test_t3_r3_sin_pullback_es_not_triggered():
    """Consolidacion con S_b tocado pero SIN volumen bajo en ninguna
    sesion p. Se fuerza volumen ALTO en todos los p con close <= S_b+tol
    para garantizar v_norm(p) > -0.3 en todos los candidatos."""
    n = 103
    close = np.full(n, 130.0)
    close[60:100] = [118.0 if i % 2 == 0 else 130.0 for i in range(40)]
    close[100] = 135.0
    volume = _noisy_volume(n)
    # Forzar volumen ALTO en los p con close=118 (candidatos a pullback)
    for i in range(60, 100):
        if close[i] <= 118.6:
            volume[i] = 3_000_000.0
    df = _df_from_closes(close, volume)
    status, p_pos, v = _r3_pullback(df["Close"], df["Volume"], 100, 118.0, 0.6)
    assert status == NOT_TRIGGERED, (
        f"status={status} p_pos={p_pos} v={v} (esperado NOT_TRIGGERED: "
        f"volumen alto en todos los p)"
    )


# ----------------- T4: R4 confirmacion -----------------

def test_t4_r4_falla_si_b_plus_1_no_supera():
    closes = np.array([100.0] * 50)
    closes[40] = 110.0
    closes[41] = 100.0
    closes[42] = 110.0
    df = _df_from_closes(closes)
    status = _r4_confirmation(df["Close"], 40, R_b=100.0, tol_b=0.6)
    assert status == NOT_TRIGGERED


def test_t4_r4_pass_con_tres_cierres():
    closes = np.array([100.0] * 50)
    closes[40] = 105.0
    closes[41] = 106.0
    closes[42] = 107.0
    df = _df_from_closes(closes)
    status = _r4_confirmation(df["Close"], 40, R_b=100.0, tol_b=0.6)
    assert status == PASS


# ----------------- T5: R5 volumen ruptura -----------------

def test_t5_r5_volumen_bajo_es_not_triggered():
    """b=60, ventana [20, 59] con varianza, V(60) bajo."""
    n = 80
    volume = _noisy_volume(n)
    volume[60] = 200_000.0  # bajo respecto a la media ~1M
    df = _df_from_closes(np.full(n, 100.0), volume)
    status, v = _r5_breakout_volume(df["Volume"], 60)
    assert status == NOT_TRIGGERED, f"status={status} v={v}"
    assert v is not None and v < 0.5


def test_t5_r5_volumen_alto_es_pass():
    n = 80
    volume = _noisy_volume(n)
    volume[60] = 3_000_000.0
    df = _df_from_closes(np.full(n, 100.0), volume)
    status, v = _r5_breakout_volume(df["Volume"], 60)
    assert status == PASS
    assert v >= 0.5


# ----------------- T6: aislamiento -----------------

FORBIDDEN_PREFIXES = [
    "src.pipeline",
    "indicators.wyckoff_orchestrator",
    "indicators.wyckoff_distribution",
    "indicators.wyckoff_accumulation",
]


def test_t6_ast_sin_imports_prohibidos():
    src = pathlib.Path("scripts/audit/reaccumulation_detector.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for pref in FORBIDDEN_PREFIXES:
                assert not mod.startswith(pref), f"forbidden import: {mod}"


def test_t6_ast_sin_call_a_classify():
    """Verifica por AST que el detector NO llama a classify_*; ignora
    docstrings y comentarios."""
    src = pathlib.Path("scripts/audit/reaccumulation_detector.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    forbidden_calls = {"classify_wyckoff_phase", "classify_phase", "classify_phase_at"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            assert name not in forbidden_calls, f"forbidden call: {name}"


def test_t6_detect_no_escribe_ficheros(monkeypatch):
    import scripts.audit.reaccumulation_detector as det
    df = _make_t7_series()
    original_open = open
    writes = []
    def spy_open(path, mode="r", *a, **kw):
        if "w" in mode or "a" in mode:
            writes.append(str(path))
        return original_open(path, mode, *a, **kw)
    monkeypatch.setattr("builtins.open", spy_open)
    det.detect_reaccumulation(df, "TEST")
    assert writes == [], f"detect_reaccumulation escribio: {writes}"


# ----------------- T7: end-to-end evento positivo -----------------

def test_t7_serie_sintetica_produce_contexto_up_en_b_menos_60():
    df = _make_t7_series()
    trend = _trend_component(df, "TEST")
    ctx_pos = 540 - L_PREV
    assert not pd.isna(trend.iloc[ctx_pos])
    assert float(trend.iloc[ctx_pos]) >= TREND_UP_THRESHOLD, (
        f"contexto no UP en b-60: trend={trend.iloc[ctx_pos]}"
    )


def test_t7_evento_confirmado_exactamente_en_b_plus_2():
    df = _make_t7_series()
    eps = detect_reaccumulation(df, "TEST")
    assert len(eps) >= 1, "no se detecto episodio en T7"
    ep = [e for e in eps if e["date_start"] == str(df.index[540].date())]
    assert len(ep) == 1
    ep = ep[0]
    assert ep["date_confirmed"] == str(df.index[542].date())
    assert ep["r1_res"] == PASS
    assert ep["r2_res"] == PASS
    assert ep["r3_res"] == PASS
    assert ep["r4_res"] == PASS
    assert ep["r5_res"] == PASS


def test_t7_sin_evento_antes_de_b_plus_2():
    df = _make_t7_series()
    eps = detect_reaccumulation(df, "TEST")
    for e in eps:
        assert e["date_confirmed"] >= str(df.index[542].date())