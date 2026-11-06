# -*- coding: utf-8 -*-
"""Tests de stale_context para Sector Breadth (S30, 2026-11-06)."""
from unittest.mock import patch

import pandas as pd
import pytest

from src.pipeline.breadth_metrics import (
    _compute_stale_context,
    compute_breadth_metrics,
)
from src.report.sectorial import render_sector_breadth


class TestComputeStaleContext:

    def test_df_vacio_devuelve_none(self):
        assert _compute_stale_context(pd.DataFrame(), pd.Timestamp.now()) is None

    def test_reference_date_none_devuelve_none(self):
        df = pd.DataFrame({"x": [1]})
        assert _compute_stale_context(df, None) is None

    def test_selecciona_worst_market(self):
        df = pd.DataFrame({"x": [1]})
        fake = {
            "US_EQUITY": {"n": 100, "coverage_at_session": 0.98, "status": "VALID"},
            "LSE": {"n": 86, "coverage_at_session": 0.186, "status": "VALID_WITH_MISSING"},
            "BME": {"n": 34, "coverage_at_session": 0.55, "status": "VALID_WITH_MISSING"},
        }
        with patch("src.utils._compute_by_market", return_value=fake):
            ctx = _compute_stale_context(df, pd.Timestamp.now())
        assert ctx is not None
        assert ctx["worst_market"] == "LSE"
        assert ctx["worst_coverage"] == pytest.approx(0.186)

    def test_filtra_skip_invalid_pending(self):
        df = pd.DataFrame({"x": [1]})
        fake = {
            "US_EQUITY": {"n": 100, "coverage_at_session": 0.98, "status": "VALID"},
            "UNKNOWN": {"n": 3, "coverage_at_session": None, "status": "INVALID"},
            "XETRA": {"n": 34, "coverage_at_session": None, "status": "SKIP"},
        }
        with patch("src.utils._compute_by_market", return_value=fake):
            ctx = _compute_stale_context(df, pd.Timestamp.now())
        assert ctx is not None
        assert ctx["worst_market"] == "US_EQUITY"

    def test_todos_filtrados_devuelve_none(self):
        df = pd.DataFrame({"x": [1]})
        fake = {
            "UNKNOWN": {"n": 3, "coverage_at_session": None, "status": "INVALID"},
        }
        with patch("src.utils._compute_by_market", return_value=fake):
            assert _compute_stale_context(df, pd.Timestamp.now()) is None

    @pytest.mark.parametrize("worst", ["LSE", "EURONEXT", "XETRA", "BME"])
    def test_selecciona_cualquier_worst_market(self, worst):
        """El mecanismo no hardcodea LSE: cualquier mercado puede ser worst."""
        df = pd.DataFrame({"x": [1]})
        fake = {
            "US_EQUITY": {"n": 100, "coverage_at_session": 0.98, "status": "VALID"},
            worst: {"n": 50, "coverage_at_session": 0.10, "status": "VALID_WITH_MISSING"},
        }
        with patch("src.utils._compute_by_market", return_value=fake):
            ctx = _compute_stale_context(df, pd.Timestamp.now())
        assert ctx is not None
        assert ctx["worst_market"] == worst
        assert ctx["worst_coverage"] == pytest.approx(0.10)


class TestComputeBreadthMetricsPropagaContexto:

    def test_propaga_contexto_en_data_pending(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "outputs" / "history").mkdir(parents=True)
        fake_ctx = {"worst_market": "LSE", "worst_coverage": 0.186,
                    "expected_session": "2026-11-05"}
        with patch(
                "src.pipeline.breadth_metrics._compute_sector_breadth_health",
                return_value=(None, True, "DATA_PENDING")), patch(
                "src.pipeline.breadth_metrics._compute_stale_context",
                return_value=fake_ctx), patch(
                "src.pipeline.breadth_metrics._compute_momentum_amplitud",
                return_value=None):
            out = compute_breadth_metrics(
                df_stocks=pd.DataFrame({"x": [1]}),
                df_market=pd.DataFrame(),
                holdings_df=pd.DataFrame(),
                reference_date=pd.Timestamp("2026-11-06"),
            )
        assert out["sector_breadth_stale_context"] == fake_ctx

    def test_no_computa_contexto_si_no_data_pending(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "outputs" / "history").mkdir(parents=True)
        with patch(
                "src.pipeline.breadth_metrics._compute_sector_breadth_health",
                return_value=(None, False, None)), patch(
                "src.pipeline.breadth_metrics._compute_stale_context",
                side_effect=AssertionError("no debe llamarse")), patch(
                "src.pipeline.breadth_metrics._compute_momentum_amplitud",
                return_value=None):
            out = compute_breadth_metrics(
                df_stocks=pd.DataFrame({"x": [1]}),
                df_market=pd.DataFrame(),
                holdings_df=pd.DataFrame(),
                reference_date=pd.Timestamp("2026-11-06"),
            )
        assert out["sector_breadth_stale_context"] is None


class TestRenderConContexto:

    def _snap(self):
        return pd.DataFrame([{
            "date": "2026-11-05",
            "sector": "XLK",
            "pct_above_ema20": 0.5, "pct_above_ema50": 0.5,
            "pct_above_ema200": 0.5, "pct_rs_positive": 0.5,
            "pct_momentum_positive": 0.5,
            "new_highs": 1, "new_lows": 1,
            "advances": 5, "declines": 5, "ad_net": 0,
            "n_valid_ema200": 20, "n_total": 20, "n_valid_nhnl": 20,
        }])

    @pytest.mark.parametrize("worst, cov", [
        ("LSE", 0.186),
        ("EURONEXT", 0.34),
        ("XETRA", 0.425),
        ("BME", 0.55),
    ])
    def test_stale_context_cualquier_mercado(self, worst, cov):
        """El render publica cualquier mercado del contexto, no solo LSE."""
        lines = render_sector_breadth(
            self._snap(), is_stale=True, stale_reason="DATA_PENDING",
            stale_context={"worst_market": worst, "worst_coverage": cov},
        )
        joined = "".join(lines)
        assert worst in joined
        assert f"{cov:.1%}" in joined
        assert "NYSE pendiente" not in joined

    def test_sin_contexto_mantiene_texto_legacy(self):
        lines = render_sector_breadth(
            self._snap(), is_stale=True, stale_reason="DATA_PENDING",
        )
        joined = "".join(lines)
        assert "NYSE pendiente" in joined
        assert "stock_prices" in joined

    def test_contexto_sin_worst_market_cae_a_legacy(self):
        lines = render_sector_breadth(
            self._snap(), is_stale=True, stale_reason="DATA_PENDING",
            stale_context={},
        )
        joined = "".join(lines)
        assert "NYSE pendiente" in joined


class TestEndToEnd:
    """Pipeline: _compute_by_market -> stale_context -> metrics -> render.

    Sin hardcode: cualquier mercado con cobertura baja aparece en el render
    del Sector Breadth, no solo LSE (que era el unico caso del fix S30).
    """

    def _snap(self):
        return pd.DataFrame([{
            "date": "2026-11-05",
            "sector": "XLK",
            "pct_above_ema20": 0.5, "pct_above_ema50": 0.5,
            "pct_above_ema200": 0.5, "pct_rs_positive": 0.5,
            "pct_momentum_positive": 0.5,
            "new_highs": 1, "new_lows": 1,
            "advances": 5, "declines": 5, "ad_net": 0,
            "n_valid_ema200": 20, "n_total": 20, "n_valid_nhnl": 20,
        }])

    @pytest.mark.parametrize("worst", ["EURONEXT", "XETRA", "BME", "LSE"])
    def test_pipeline_completo_cualquier_worst_market(
            self, worst, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "outputs" / "history").mkdir(parents=True)
        fake_by_market = {
            "US_EQUITY": {"n": 100, "coverage_at_session": 0.98, "status": "VALID"},
            worst: {"n": 50, "coverage_at_session": 0.20, "status": "VALID_WITH_MISSING"},
        }
        with patch("src.utils._compute_by_market", return_value=fake_by_market), \
             patch(
                "src.pipeline.breadth_metrics._compute_sector_breadth_health",
                return_value=(None, True, "DATA_PENDING")), \
             patch(
                "src.pipeline.breadth_metrics._compute_momentum_amplitud",
                return_value=None):
            out = compute_breadth_metrics(
                df_stocks=pd.DataFrame({"x": [1]}),
                df_market=pd.DataFrame(),
                holdings_df=pd.DataFrame(),
                reference_date=pd.Timestamp("2026-11-06"),
            )
        ctx = out["sector_breadth_stale_context"]
        assert ctx is not None
        assert ctx["worst_market"] == worst

        lines = render_sector_breadth(
            self._snap(), is_stale=True, stale_reason="DATA_PENDING",
            stale_context=ctx,
        )
        joined = "".join(lines)
        assert worst in joined
        assert "20.0%" in joined
        assert "NYSE pendiente" not in joined
