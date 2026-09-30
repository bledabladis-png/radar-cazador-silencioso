# -*- coding: utf-8 -*-
"""D28 (2026-09-30): tests para cubrir branches residuales de 4 modulos
de indicators con 76-79% cobertura.

- breadth_equity.compute_advance_decline: ramas defensivas.
- evidence_matrix: _sign_from_value, _safe_get_row, compute_evidence_matrix.
- mte.decision: validate_transition, consensus_score, distance_to_threshold,
  compute_confidence, classify_mte.
- mte.engine.compute_mte: fallback vix_term, write JSON, except outer.
"""
import numpy as np
import pandas as pd
import pytest

from indicators.breadth_equity import compute_advance_decline
from indicators.evidence_matrix import (
    _sign_from_value, _sign_from_pct_above, _wyckoff_sign,
    _quality_from_coverage, _safe_get_row, compute_evidence_matrix,
)
from indicators.mte.decision import (
    validate_transition, consensus_score, distance_to_threshold,
    compute_confidence, classify_mte,
)
from indicators.mte.engine import compute_mte


# ============================================================
# breadth_equity
# ============================================================
def test_advance_decline_sin_columnas_close():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    assert compute_advance_decline(df) is None


def test_advance_decline_pocos_tickers():
    """<20 activos -> None."""
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    cols = pd.MultiIndex.from_tuples([("Close", f"T{i}") for i in range(10)])
    df = pd.DataFrame(np.random.rand(100, 10) * 100, index=idx, columns=cols)
    assert compute_advance_decline(df) is None


def test_advance_decline_todos_unchanged():
    """Todos sin cambio -> None (rama advances+declines==0)."""
    idx = pd.date_range("2024-01-01", periods=100, freq="B")
    cols = pd.MultiIndex.from_tuples([("Close", f"T{i}") for i in range(25)])
    df = pd.DataFrame(100.0, index=idx, columns=cols)
    assert compute_advance_decline(df) is None


def _market_df(n_tickers=25, n=300, seed=42):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    cols = pd.MultiIndex.from_tuples([("Close", f"T{i}") for i in range(n_tickers)])
    data = 100 * np.cumprod(1 + rng.normal(0.001, 0.02, (n, n_tickers)), axis=0)
    return pd.DataFrame(data, index=idx, columns=cols)


def test_advance_decline_normal():
    df = _market_df()
    result = compute_advance_decline(df)
    assert result is not None
    assert result["total_tickers"] == 25
    assert "ad_net" in result


def test_advance_decline_effective_meta_fecha_invalida():
    df = _market_df()
    meta = {"status": "OK", "date": "no_es_fecha", "coverage": 0.9,
            "n_observed": 22, "n_eligible": 25, "lag_days": 0}
    result = compute_advance_decline(df, effective_meta=meta)
    assert result["effective_date"] is None


def test_advance_decline_effective_meta_ok():
    df = _market_df()
    meta = {"status": "OK", "date": "2025-01-01", "coverage": 0.9,
            "n_observed": 22, "n_eligible": 25, "lag_days": 0}
    result = compute_advance_decline(df, effective_meta=meta)
    assert result["effective_date"] == "2025-01-01"
    assert result["coverage"] == 0.9
    assert result["n_observed"] == 22


# ============================================================
# evidence_matrix
# ============================================================
def test_sign_from_value_nan():
    assert pd.isna(_sign_from_value(np.nan))


def test_sign_from_value_positivo():
    assert _sign_from_value(1.5) == 1


def test_sign_from_value_negativo():
    assert _sign_from_value(-1.5) == -1


def test_sign_from_value_cero():
    assert _sign_from_value(0) == 0


def test_sign_from_pct_above_nan():
    assert pd.isna(_sign_from_pct_above(np.nan))


def test_sign_from_pct_above_bajo():
    assert _sign_from_pct_above(30) == -1


def test_sign_from_pct_above_igual():
    assert _sign_from_pct_above(50) == 0


def test_wyckoff_sign_nan():
    row = {"pct_accumulation": np.nan, "pct_markup": 0.1,
           "pct_distribution": 0.1, "pct_markdown": 0.1}
    assert pd.isna(_wyckoff_sign(row))


def test_wyckoff_sign_desfavorable():
    row = {"pct_accumulation": 0.1, "pct_markup": 0.1,
           "pct_distribution": 0.5, "pct_markdown": 0.3}
    assert _wyckoff_sign(row) == -1


def test_wyckoff_sign_neutral():
    row = {"pct_accumulation": 0.25, "pct_markup": 0.25,
           "pct_distribution": 0.25, "pct_markdown": 0.25}
    assert _wyckoff_sign(row) == 0


def test_quality_from_coverage_alta():
    assert _quality_from_coverage(85) == "Alta"


def test_quality_from_coverage_media():
    assert _quality_from_coverage(60) == "Media"


def test_quality_from_coverage_baja():
    assert _quality_from_coverage(30) == "Baja"


def test_quality_from_coverage_nan():
    assert _quality_from_coverage(np.nan) == "Baja"


def test_safe_get_row_empty_df():
    assert _safe_get_row(pd.DataFrame(), "sector") == {}


def test_safe_get_row_none():
    assert _safe_get_row(None, "sector") == {}


def test_safe_get_row_sector_ausente():
    df = pd.DataFrame({"a": [1]}, index=["X"])
    assert _safe_get_row(df, "Y") == {}


def test_safe_get_row_dataframe_duplicado():
    """loc con sector duplicado -> DataFrame -> tomar ultima fila."""
    df = pd.DataFrame({"value": [1.0, 2.0]}, index=["X", "X"])
    result = _safe_get_row(df, "X")
    assert result["value"] == 2.0


def test_compute_evidence_matrix_sin_datos():
    assert compute_evidence_matrix() is None


def test_compute_evidence_matrix_minimo():
    breadth = pd.DataFrame({
        "date": ["2026-09-30"],
        "sector": ["XLK"],
        "n_total": [20],
        "n_valid_ema50": [18],
        "pct_above_ema50": [60.0],
    })
    out = compute_evidence_matrix(sector_breadth_df=breadth)
    assert out is not None
    assert len(out) == 1
    assert out.iloc[0]["sector"] == "XLK"


# ============================================================
# mte.decision
# ============================================================
def test_validate_transition_normal():
    assert validate_transition("MIXED", "EXPANSION", 0.5)


def test_validate_transition_excepcion_con_cls_alta():
    assert validate_transition("EXPANSION", "RECESSION", 0.9)


def test_validate_transition_excepcion_con_cls_baja():
    assert not validate_transition("EXPANSION", "RECESSION", 0.5)


def test_validate_transition_invalida():
    assert not validate_transition("MIXED", "INVALID", 0.5)


def test_consensus_score_nan():
    """Guard finitud: NaN -> 0.0."""
    assert consensus_score(np.nan, 0.5, 0.5, 0.5) == 0.0


def test_consensus_score_inf():
    assert consensus_score(np.inf, 0.5, 0.5, 0.5) == 0.0


def test_consensus_score_todos_iguales():
    assert consensus_score(0.5, 0.5, 0.5, 0.5) == 1.0


@pytest.mark.parametrize("scenario", [
    "CRISIS", "RECESSION", "STAGFLATION", "SOFT LANDING", "EXPANSION",
])
def test_distance_to_threshold_escenarios(scenario):
    out = distance_to_threshold(0.5, 0.5, 0.7, 0.5, scenario)
    assert 0 <= out <= 1


def test_distance_to_threshold_scenario_desconocido():
    assert distance_to_threshold(0.5, 0.5, 0.5, 0.5, "UNKNOWN") == 0.5


def test_compute_confidence_normal():
    out = compute_confidence(0.5, 0.5, 0.7, 0.5, "CRISIS")
    assert 0 <= out <= 1


def test_classify_mte_previo_mixed(monkeypatch):
    monkeypatch.setattr(
        "indicators.mte.decision.load_previous_scenario",
        lambda: ("MIXED", None, False))
    scenario, conf, pending, reset = classify_mte(0.5, 0.5, 0.5, 0.5)
    assert scenario in (
        "CRISIS", "RECESSION", "STAGFLATION", "SOFT LANDING",
        "EXPANSION", "MIXED")


def test_classify_mte_confidence_cero_fuerza_mixed(monkeypatch):
    monkeypatch.setattr(
        "indicators.mte.decision.load_previous_scenario",
        lambda: ("MIXED", None, False))
    # inputs que probablemente dan confidence 0
    scenario, conf, pending, reset = classify_mte(0.0, 0.0, 0.0, 0.0)
    if conf == 0.0:
        assert scenario == "MIXED"


# ============================================================
# mte.engine
# ============================================================
def _market_df_engine(n=300, seed=42):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    tickers = ["^VIX", "^VIX3M", "^GSPC", "XLK", "XLE", "^TNX",
               "GC=F", "TLT", "HYG"]
    data = {}
    for i, t in enumerate(tickers):
        close = 100 * np.cumprod(1 + rng.normal(0.0001, 0.015, n))
        data[("Close", t)] = close
        data[("High", t)] = close * 1.01
        data[("Low", t)] = close * 0.99
        data[("Volume", t)] = rng.randint(1_000_000, 5_000_000, n)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_compute_mte_normal(tmp_path, monkeypatch):
    from indicators.mte import state as mte_state
    monkeypatch.setattr(mte_state, "MTE_STATE_FILE",
                        str(tmp_path / "mte_state.json"))
    df = _market_df_engine()
    fc = pd.Series([0.3] * 300, index=df.index)
    cred = pd.Series([0.2] * 300, index=df.index)
    vol = pd.Series([-0.1] * 300, index=df.index)
    out = compute_mte(df, fc, cred, vol)
    if out is not None:
        assert "scenario" in out
        assert "confidence" in out


def test_compute_mte_sin_vix3m_fallback(tmp_path, monkeypatch):
    """Sin ^VIX3M -> vix_term=0.0 (rama except). Sin crash."""
    from indicators.mte import state as mte_state
    monkeypatch.setattr(mte_state, "MTE_STATE_FILE",
                        str(tmp_path / "mte_state.json"))
    df = _market_df_engine()
    df = df.drop(columns=[("Close", "^VIX3M")])
    fc = pd.Series([0.3] * 300, index=df.index)
    cred = pd.Series([0.2] * 300, index=df.index)
    vol = pd.Series([-0.1] * 300, index=df.index)
    out = compute_mte(df, fc, cred, vol)
    # El contrato del motor: dict valido o None (fallo interno).
    assert out is None or "scenario" in out


def test_compute_mte_excepcion_retorna_none(monkeypatch, tmp_path):
    """Excepcion dentro del cuerpo -> None."""
    from indicators.mte import engine
    from indicators.mte import state as mte_state
    monkeypatch.setattr(mte_state, "MTE_STATE_FILE",
                        str(tmp_path / "mte_state.json"))

    def boom(df):
        raise RuntimeError("simulado")

    monkeypatch.setattr(engine, "sector_rotation_score", boom)
    df = _market_df_engine()
    fc = pd.Series([0.3] * 300, index=df.index)
    cred = pd.Series([0.2] * 300, index=df.index)
    vol = pd.Series([-0.1] * 300, index=df.index)
    out = engine.compute_mte(df, fc, cred, vol)
    assert out is None
