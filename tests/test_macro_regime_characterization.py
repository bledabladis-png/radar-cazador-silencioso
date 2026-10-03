# -*- coding: utf-8 -*-
"""Caracterizacion de regimes/macro_regime.py (frente B, 2026-10-03).

Cubre las 12 ramas de la cadena if/elif/else de compute_macro_regime,
el orden de precedencia, y el contrato de ramas inalcanzables sin
fundamentales.

Patron: monkeypatch de compute_macro_signals y compute_macro_score en
el namespace de regimes.macro_regime. NO se construye df_market real
(fragil: 20+ columnas mantenidas a mano). Justificado en
docs/auditoria/s03_deep/macro_regime.md y en el docstring de
test_macro_regime_confidence.py.

Contrato: las ramas 4 (INFLATION SHOCK), 5 (STAGFLATION) y 10
(DEFLATION) dependen de last_inflation. Sin df_macro_manual, la
columna 'inflation' no existe en all_signals y
all_signals.get('inflation', pd.Series(0)).iloc[-1] == 0. Por tanto,
esas ramas no disparan. Comportamiento por diseno (N/D, no imputacion).
"""
import pytest
import pandas as pd

from regimes import macro_regime as mr


def _all_signals(n=3, volatility=0.0, credit=0.0, market_strength=0.0,
                 liquidity=0.0, curve=0.0, inflation=None):
    """all_signals sintetico con valores constantes en la ultima fecha."""
    idx = pd.date_range("2026-01-01", periods=n)
    data = {
        "volatility": [volatility] * n,
        "credit": [credit] * n,
        "market_strength": [market_strength] * n,
        "liquidity": [liquidity] * n,
        "curve": [curve] * n,
    }
    if inflation is not None:
        data["inflation"] = [inflation] * n
    return pd.DataFrame(data, index=idx)
def _macro_score(n=3, last=0.0):
    """Serie sintetica de macro_score con valor constante."""
    idx = pd.date_range("2026-01-01", periods=n)
    return pd.Series([last] * n, index=idx)


def _run(monkeypatch, *, last=0.0, volatility=0.0, credit=0.0,
         market_strength=0.0, liquidity=0.0, curve=0.0, inflation=None):
    """Invoca compute_macro_regime con all_signals y macro_score sinteticos."""
    signals = _all_signals(
        volatility=volatility, credit=credit,
        market_strength=market_strength, liquidity=liquidity,
        curve=curve, inflation=inflation,
    )
    score = _macro_score(last=last)
    monkeypatch.setattr(mr, "compute_macro_signals", lambda *a, **k: signals)
    monkeypatch.setattr(mr, "compute_macro_score", lambda s: score)
    _, regime, _, _ = mr.compute_macro_regime(
        df_market=None, df_macro_manual=None,
        liquidity_score=None, vol_score=None,
    )
    return regime


RAMA_CASES = [
    # (id, kwargs, regime_esperado)
    ("rama1_vol_crisis",     {"volatility": -2.5}, "LIQUIDITY CRISIS"),
    ("rama2_vol_credit",     {"volatility": -1.6, "credit": -0.6}, "LIQUIDITY CRISIS"),
    ("rama3_recession",      {"last": -0.5, "market_strength": -0.6}, "RECESSION"),
    ("rama4_infl_shock",     {"inflation": 0.5, "market_strength": -0.1}, "INFLATION SHOCK"),
    ("rama5_stagflation",    {"last": -0.3, "inflation": 0.5}, "STAGFLATION"),
    ("rama6_goldilocks",     {"last": 0.3, "market_strength": 0.3, "inflation": -0.2, "volatility": -0.1}, "GOLDILOCKS"),
    ("rama7_expansion",      {"last": 0.1, "market_strength": 0.1, "liquidity": 0.0, "volatility": -0.1, "curve": 0.1}, "EXPANSION"),
    ("rama8_late_expansion", {"last": 0.3, "inflation": 0.3}, "LATE EXPANSION"),
    ("rama9_recovery",       {"last": 0.01, "market_strength": 0.1}, "RECOVERY"),
    ("rama10_deflation",     {"last": 0.1, "inflation": -0.6}, "DEFLATION"),
    ("rama11_slowdown",      {"last": -0.3, "market_strength": -0.1}, "SLOWDOWN"),
    ("rama12_mixed",         {"last": 0.0, "market_strength": 0.0}, "MIXED"),
]
@pytest.mark.parametrize(
    "name,kwargs,expected", RAMA_CASES,
    ids=[c[0] for c in RAMA_CASES],
)
def test_rama_regime(monkeypatch, name, kwargs, expected):
    """Cada rama de la cadena if/elif/else se alcanza con su input."""
    assert _run(monkeypatch, **kwargs) == expected


PRECEDENCE_CASES = [
    # input que satisface 2+ ramas; gana la de menor indice.
    ("vol_crisis_gana_a_vol_credit",
     {"volatility": -2.5, "credit": -0.6},
     "LIQUIDITY CRISIS"),
    ("vol_credit_gana_a_recession",
     {"volatility": -1.6, "credit": -0.6, "last": -0.5, "market_strength": -0.6},
     "LIQUIDITY CRISIS"),
    ("recession_gana_a_slowdown",
     {"last": -0.5, "market_strength": -0.6},
     "RECESSION"),
    ("goldilocks_gana_a_expansion",
     {"last": 0.3, "market_strength": 0.3, "inflation": -0.2,
      "volatility": -0.1, "liquidity": 0.0, "curve": 0.1},
     "GOLDILOCKS"),
    ("late_expansion_gana_a_recovery",
     {"last": 0.3, "market_strength": 0.1, "inflation": 0.3},
     "LATE EXPANSION"),
]


@pytest.mark.parametrize(
    "name,kwargs,expected", PRECEDENCE_CASES,
    ids=[c[0] for c in PRECEDENCE_CASES],
)
def test_precedencia_ramas(monkeypatch, name, kwargs, expected):
    """Con multiples ramas aplicables, gana la de menor indice."""
    assert _run(monkeypatch, **kwargs) == expected
def test_sin_fundamentales_ramas_inflacion_no_disparan(monkeypatch):
    """Contrato: sin df_macro_manual, last_infl=0.

    Ramas 4 (INFLATION SHOCK), 5 (STAGFLATION) y 10 (DEFLATION)
    dependen de last_inflation y no disparan sin fundamentales. El
    caso 'intento de rama 4' con market_strength=-0.5 y last=0.5
    cae a MIXED: la cadena no encuentra ninguna rama aplicable.
    """
    regime = _run(monkeypatch, inflation=None, market_strength=-0.5, last=0.5)
    assert regime != "INFLATION SHOCK"
    assert regime != "STAGFLATION"
    assert regime != "DEFLATION"
    assert regime == "MIXED"
# ---------------------------------------------------------------------------
# Bloque 2: compute_macro_score y weighted_score (puras, sin monkeypatch).
# ---------------------------------------------------------------------------
# Nota de diseno (candidato MR-8, MEDIA): el comentario de
# compute_macro_score dice que si un nivel entero no tiene datos su
# peso se redistribuye entre los niveles con evidencia. En la practica
# NO ocurre: weighted_score devuelve pd.Series(0) (no NaN) cuando
# available=[] o cuando todos sus componentes son NaN, y el bucle de
# nivel superior solo mira s.notna(). Resultado:
#   score = (Wc*c + Wi*i + Wx*x) / (Wc+Wi+Wx)
# con c/i/x=0 cuando su nivel no tiene evidencia. La renormalizacion
# SI ocurre DENTRO de un nivel (por componente). Estos tests anclan
# el comportamiento actual del codigo, no el declarado en el comentario.
from config.weights import LEVEL_WEIGHTS

_WC = LEVEL_WEIGHTS["critical"]
_WI = LEVEL_WEIGHTS["important"]
_WX = LEVEL_WEIGHTS["contextual"]


def _only_critical_factor():
    """Peso relativo del nivel critical cuando solo el tiene datos."""
    return _WC / (_WC + _WI + _WX)


def _signals_const(n=3, **cols):
    """DataFrame n filas con columnas constantes. NaN via float('nan')."""
    idx = pd.date_range("2026-01-01", periods=n)
    return pd.DataFrame({k: [v] * n for k, v in cols.items()}, index=idx)


def test_macro_score_todo_nan_es_cero():
    """Sin ninguna senal valida, todos los niveles valen 0 -> score 0."""
    s = _signals_const(
        curve=float("nan"), credit=float("nan"), volatility=float("nan"),
        liquidity=float("nan"), real_liquidity=float("nan"),
        dollar=float("nan"), commodities=float("nan"), breadth=float("nan"),
        market_strength=float("nan"),
    )
    out = mr.compute_macro_score(s)
    assert (out == 0.0).all()


def test_macro_score_solo_critico_disponible():
    """Solo critical valido: score = Wc/(Wc+Wi+Wx) * 0.5.

    No redistribuye peso de important/contextual (ver nota MR-8).
    """
    s = _signals_const(
        curve=0.5, credit=float("nan"), volatility=float("nan"),
        liquidity=float("nan"), real_liquidity=float("nan"),
        dollar=float("nan"), commodities=float("nan"), breadth=float("nan"),
        market_strength=float("nan"),
    )
    out = mr.compute_macro_score(s)
    assert out.iloc[-1] == pytest.approx(_only_critical_factor() * 0.5)


def test_macro_score_solo_important_disponible():
    """Solo important valido: score = Wi/(Wc+Wi+Wx) * 0.4."""
    s = _signals_const(
        curve=float("nan"), credit=float("nan"), volatility=float("nan"),
        liquidity=float("nan"), real_liquidity=float("nan"),
        dollar=0.4, commodities=0.4, breadth=0.4,
        market_strength=float("nan"),
    )
    out = mr.compute_macro_score(s)
    factor = _WI / (_WC + _WI + _WX)
    assert out.iloc[-1] == pytest.approx(factor * 0.4)


def test_macro_score_mezcla_fundamentales_solo_donde_hay_datos():
    """0.5*macro + 0.5*fund_mean en filas con fundamentales; macro puro en el resto."""
    idx = pd.date_range("2026-01-01", periods=3)
    s = pd.DataFrame({
        "curve": [1.0, 1.0, 1.0],
        "credit": [float("nan")] * 3,
        "volatility": [float("nan")] * 3,
        "liquidity": [float("nan")] * 3,
        "real_liquidity": [float("nan")] * 3,
        "dollar": [float("nan")] * 3,
        "commodities": [float("nan")] * 3,
        "breadth": [float("nan")] * 3,
        "market_strength": [float("nan")] * 3,
        "inflation": [float("nan"), float("nan"), 0.0],
    }, index=idx)
    out = mr.compute_macro_score(s)
    base = _only_critical_factor() * 1.0
    # Fila 0-1: macro puro = base. Fila 2: 0.5*base + 0.5*0 = base/2.
    # rolling(2, min_periods=1): [base, base, 0.75*base].
    assert out.iloc[0] == pytest.approx(base)
    assert out.iloc[2] == pytest.approx(0.75 * base)


def test_macro_score_rolling_dos_periodos():
    """rolling(2, min_periods=1).mean() suaviza con ventana de 2."""
    idx = pd.date_range("2026-01-01", periods=3)
    s = pd.DataFrame({
        "curve": [0.0, 0.4, 0.8],
        "credit": [float("nan")] * 3,
        "volatility": [float("nan")] * 3,
        "liquidity": [float("nan")] * 3,
        "real_liquidity": [float("nan")] * 3,
        "dollar": [float("nan")] * 3,
        "commodities": [float("nan")] * 3,
        "breadth": [float("nan")] * 3,
        "market_strength": [float("nan")] * 3,
    }, index=idx)
    out = mr.compute_macro_score(s)
    f = _only_critical_factor()
    b0 = f * 0.0
    b1 = f * 0.4
    b2 = f * 0.8
    assert out.iloc[0] == pytest.approx(b0)
    assert out.iloc[1] == pytest.approx((b0 + b1) / 2)
    assert out.iloc[2] == pytest.approx((b1 + b2) / 2)

# ---------------------------------------------------------------------------
# Bloque 3: cadena de fallback de 'volatility' en compute_macro_signals.
# Cubre tambien MR-2 (audit doc S-03-deep): compute_macro_regime accede a
# all_signals['volatility'] sin guard. Caso D (sin VIX ni vol_regime_score)
# deja el DataFrame sin columna 'volatility' y compute_macro_regime lanza
# KeyError. WONT FIX condicional documentado; aqui queda con evidencia.
# ---------------------------------------------------------------------------
import indicators.breadth as _breadth_mod


def _fake_market_df(n=3):
    idx = pd.date_range("2026-01-01", periods=n)
    return pd.DataFrame(index=idx)


def _fake_get_col(series_map):
    def _impl(df, ticker, col):
        if ticker in series_map:
            return series_map[ticker]
        raise KeyError(ticker)
    return _impl


def _mock_helpers(monkeypatch, series_map):
    """Sustituye get_col, tanh_normalize y helpers para aislar volatilidad."""
    monkeypatch.setattr(mr, "get_col", _fake_get_col(series_map))
    monkeypatch.setattr(mr, "tanh_normalize", lambda s: s)
    monkeypatch.setattr(mr, "momentum_score", lambda s, w: s)
    monkeypatch.setattr(mr, "normalize_momentum", lambda s: s)
    monkeypatch.setattr(
        mr, "credit_risk_signal",
        lambda df: pd.Series(0.0, index=df.index),
    )
    monkeypatch.setattr(
        _breadth_mod, "compute_breadth",
        lambda df: tuple(pd.Series(0.0, index=df.index) for _ in range(5)),
    )
def test_vol_fallback_usa_vix_y_vix3m(monkeypatch):
    """Caso A: ^VIX + ^VIX3M -> 0.7*vix + 0.3*term, negado."""
    idx = pd.date_range("2026-01-01", periods=3)
    vix = pd.Series([10.0, 10.0, 10.0], index=idx)
    vix3m = pd.Series([20.0, 20.0, 20.0], index=idx)
    _mock_helpers(monkeypatch, {"^VIX": vix, "^VIX3M": vix3m})
    out = mr.compute_macro_signals(
        _fake_market_df(), df_macro_manual=None,
        liquidity_score=None, vol_regime_score=None,
    )
    # vol_signal = 0.7*10 + 0.3*(20-10) = 10; volatility = -10
    assert out["volatility"].iloc[-1] == pytest.approx(-10.0)


def test_vol_fallback_usa_vol_regime_score(monkeypatch):
    """Caso B: sin ^VIX3M pero vol_regime_score dado -> -vol_regime_score."""
    idx = pd.date_range("2026-01-01", periods=3)
    vix = pd.Series([10.0, 10.0, 10.0], index=idx)
    _mock_helpers(monkeypatch, {"^VIX": vix})
    out = mr.compute_macro_signals(
        _fake_market_df(), df_macro_manual=None,
        liquidity_score=None, vol_regime_score=0.5,
    )
    assert out["volatility"].iloc[-1] == pytest.approx(-0.5)


def test_vol_fallback_solo_vix(monkeypatch):
    """Caso C: sin ^VIX3M ni vol_regime_score -> -tanh_normalize(^VIX)."""
    idx = pd.date_range("2026-01-01", periods=3)
    vix = pd.Series([10.0, 10.0, 10.0], index=idx)
    _mock_helpers(monkeypatch, {"^VIX": vix})
    out = mr.compute_macro_signals(
        _fake_market_df(), df_macro_manual=None,
        liquidity_score=None, vol_regime_score=None,
    )
    assert out["volatility"].iloc[-1] == pytest.approx(-10.0)


def test_vol_fallback_sin_vix_deja_dataframe_sin_columna(monkeypatch):
    """Caso D: sin ^VIX ni vol_regime_score -> 'volatility' ausente.

    Contrato MR-2: compute_macro_regime accede a all_signals['volatility']
    sin guard; con esta entrada lanza KeyError. WONT FIX condicional
    documentado en docs/auditoria/s03_deep/macro_regime.md.
    """
    _mock_helpers(monkeypatch, {})
    out = mr.compute_macro_signals(
        _fake_market_df(), df_macro_manual=None,
        liquidity_score=None, vol_regime_score=None,
    )
    assert "volatility" not in out.columns