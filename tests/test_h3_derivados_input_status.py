# -*- coding: utf-8 -*-
"""Centinela H3 (auditoria 2026-11-07): los derivados de Sector Breadth
deben declarar la fecha del input y su estado (VALID/DEGRADED).

Bug real: Sector Regime Matrix y Matriz de Evidencia se alimentan de
sector_breadth, que puede venir de un walk-back (DATA_PENDING en el
target_session). El auditor: 'un score puede parecer actual porque se
genera el dia 5, aunque uno de sus inputs principales tenga un dia de
retraso'.

Causa raiz: sector_breadth_stale_reason existia en run.py:181 pero
no se propagaba a compute_final_matrices, ni a los productores
(build_sector_regime_matrix, compute_evidence_matrix), ni a los
renders (render_matriz_regimen, render_matriz_evidencia).

Fix: mecanismo input_status dict {'sector_breadth': VALID|DEGRADED}.
Reutiliza indicators.mte.status. Persiste columna 'input_status' en
ambos df. Los renders publican '*Datos al YYYY-MM-DD.*' siempre y
'*[input DEGRADED] ...*' cuando aplica. Sin hardcode de modulos.

Se cubre en 3 niveles:
  1. Productor: input_status -> columna del df.
  2. Transporte: finalize construye el dict desde stale_reason.
  3. Render: lee date + input_status y publica las notas.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from indicators.mte.status import VALID as IN_VALID, DEGRADED as IN_DEGRADED
from indicators.sector_regime_matrix import build_sector_regime_matrix
from indicators.evidence_matrix import compute_evidence_matrix
from src.report.sector_context import render_matriz_regimen
from src.report.synthesis import render_matriz_evidencia


# ---------------------------------------------------------------------------
# Fixtures minimas
# ---------------------------------------------------------------------------

def _breadth_df():
    return pd.DataFrame([
        {"date": "2026-11-04", "sector": "XLK", "pct_above_ema50": 60.0,
         "n_total": 20, "n_valid_ema50": 20},
        {"date": "2026-11-04", "sector": "XLF", "pct_above_ema50": 40.0,
         "n_total": 20, "n_valid_ema50": 20},
    ])


def _flow_df():
    return pd.DataFrame([
        {"date": "2026-11-04", "sector": "XLK",
         "price_ret_20d": 0.02, "flow_20d_sum": 1e6},
        {"date": "2026-11-04", "sector": "XLF",
         "price_ret_20d": -0.01, "flow_20d_sum": -1e6},
    ])


def _sector_results():
    return pd.DataFrame([
        {"sector": "XLK", "rank": 1},
        {"sector": "XLF", "rank": 2},
    ])


def _regime_matrix_df(input_status=None, with_date=True):
    row = {
        "sector": "XLK", "price_ret_20d": 0.02, "pct_above_ema50": 60.0,
        "flow_20d_sum": 1e6, "wyckoff_phase": "MARKUP",
        "wyckoff_signal_phase": "MARKUP", "wyckoff_signal_share": 50.0,
        "wyckoff_signal_n": 5, "structure_ratio": 0.6, "n_estructuras": 10,
        "price_positive": True, "breadth_positive": True,
        "flow_positive": True, "positive_conditions": 3,
        "data_complete": True, "regime_reading": "Alineación positiva",
    }
    if with_date:
        row["date"] = "2026-11-04"
    if input_status is not None:
        row["input_status"] = input_status
    return pd.DataFrame([row])


def _evidence_matrix_df(input_status=None, with_date=True):
    row = {
        "sector": "XLK", "price_evidence": 1, "breadth_evidence": 1,
        "primary_flow_evidence": 1, "proxy_flow_evidence": 1,
        "wyckoff_evidence": 1, "credit_evidence": 0, "volatility_evidence": 0,
        "n_sector_evidence_valid": 5, "n_sector_evidence_positive": 5,
        "n_sector_evidence_negative": 0, "n_sector_evidence_neutral": 0,
        "evidence_quality": "Alta",
        "alignment_reading": "EVIDENCIA PREDOMINANTEMENTE FAVORABLE",
    }
    if with_date:
        row["date"] = "2026-11-04"
    if input_status is not None:
        row["input_status"] = input_status
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# 1. Productor: input_status -> columna
# ---------------------------------------------------------------------------

class TestProductorPersisteInputStatus:

    def test_regime_matrix_sin_input_status_es_valid(self):
        df = build_sector_regime_matrix(
            _breadth_df(), _flow_df(), _sector_results())
        assert "input_status" in df.columns
        assert (df["input_status"] == IN_VALID).all()

    def test_regime_matrix_con_degraded_propaga(self):
        df = build_sector_regime_matrix(
            _breadth_df(), _flow_df(), _sector_results(),
            input_status={"sector_breadth": IN_DEGRADED})
        assert (df["input_status"] == IN_DEGRADED).all()

    def test_evidence_matrix_sin_input_status_es_valid(self):
        df = compute_evidence_matrix(_breadth_df())
        assert "input_status" in df.columns
        assert (df["input_status"] == IN_VALID).all()

    def test_evidence_matrix_con_degraded_propaga(self):
        df = compute_evidence_matrix(
            _breadth_df(), input_status={"sector_breadth": IN_DEGRADED})
        assert (df["input_status"] == IN_DEGRADED).all()


# ---------------------------------------------------------------------------
# 2. Transporte: finalize construye dict desde stale_reason
# ---------------------------------------------------------------------------

class TestTransporte:

    def test_data_pending_devuelve_degraded(self):
        from src.pipeline.finalize import compute_final_matrices
        captured = {}
        def _fake_regime(*a, **kw):
            captured["regime"] = kw.get("input_status")
            return pd.DataFrame()
        def _fake_evidence(*a, **kw):
            captured["evidence"] = kw.get("input_status")
            return pd.DataFrame()
        with patch("indicators.sector_regime_matrix.build_sector_regime_matrix",
                   side_effect=_fake_regime), \
             patch("indicators.evidence_matrix.compute_evidence_matrix",
                   side_effect=_fake_evidence):
            compute_final_matrices(
                _breadth_df(), pd.DataFrame(), _flow_df(), pd.DataFrame(),
                _sector_results(), "X", "Y", "Z", 0.0, 0.0, 0.0,
                sector_breadth_stale_reason="DATA_PENDING")
        assert captured["regime"] == {"sector_breadth": IN_DEGRADED}
        assert captured["evidence"] == {"sector_breadth": IN_DEGRADED}

    def test_sin_stale_reason_es_valid(self):
        from src.pipeline.finalize import compute_final_matrices
        captured = {}
        def _fake(*a, **kw):
            captured["st"] = kw.get("input_status")
            return pd.DataFrame()
        with patch("indicators.sector_regime_matrix.build_sector_regime_matrix",
                   side_effect=_fake), \
             patch("indicators.evidence_matrix.compute_evidence_matrix",
                   side_effect=_fake):
            compute_final_matrices(
                _breadth_df(), pd.DataFrame(), _flow_df(), pd.DataFrame(),
                _sector_results(), "X", "Y", "Z", 0.0, 0.0, 0.0)
        assert captured["st"] == {"sector_breadth": IN_VALID}


# ---------------------------------------------------------------------------
# 3. Render: fecha + status
# ---------------------------------------------------------------------------

class TestRenderMatrizRegimen:

    def test_declara_fecha(self):
        joined = "".join(render_matriz_regimen(_regime_matrix_df()))
        assert "Datos al 2026-11-04" in joined

    def test_degraded_avisa(self):
        joined = "".join(render_matriz_regimen(
            _regime_matrix_df(input_status=IN_DEGRADED)))
        assert "[input DEGRADED]" in joined

    def test_valid_no_avisa(self):
        joined = "".join(render_matriz_regimen(
            _regime_matrix_df(input_status=IN_VALID)))
        assert "[input DEGRADED]" not in joined
        assert "Datos al 2026-11-04" in joined

    def test_sin_columna_date_no_rompe(self):
        joined = "".join(render_matriz_regimen(
            _regime_matrix_df(with_date=False)))
        assert "Datos al" not in joined
        assert "Matriz de Régimen Sectorial" in joined

    def test_sin_columna_input_status_no_rompe(self):
        joined = "".join(render_matriz_regimen(_regime_matrix_df()))
        assert "Matriz de Régimen Sectorial" in joined


class TestRenderMatrizEvidencia:

    def test_declara_fecha(self):
        joined = "".join(render_matriz_evidencia(_evidence_matrix_df()))
        assert "Datos al 2026-11-04" in joined

    def test_degraded_avisa(self):
        joined = "".join(render_matriz_evidencia(
            _evidence_matrix_df(input_status=IN_DEGRADED)))
        assert "[input DEGRADED]" in joined

    def test_valid_no_avisa(self):
        joined = "".join(render_matriz_evidencia(
            _evidence_matrix_df(input_status=IN_VALID)))
        assert "[input DEGRADED]" not in joined

    def test_sin_columna_date_no_rompe(self):
        joined = "".join(render_matriz_evidencia(
            _evidence_matrix_df(with_date=False)))
        assert "Datos al" not in joined
        assert "Matriz de Evidencia" in joined
