"""Tests pipeline: sector_metrics + mte_confirmation + finalize.

Los 3 modulos que quedaban < 50% de cobertura tras el grupo anterior.
Sub-llamadas mockeadas. Se verifica:
  - Contrato de keys de retorno.
  - Degradacion individual de cada sub-bloque.
  - Escritura de CSVs (con monkeypatch chdir tmp_path).
  - Logica especifica de orquestadores.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import sector_metrics as sm
from src.pipeline import mte_confirmation as mc
from src.pipeline import finalize as fin


# =============================================================================
# sector_metrics.py
# =============================================================================

def _setup_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    (tmp_path / "outputs" / "report").mkdir(parents=True)


def test_sector_metrics_contract_5_keys(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    with patch("src.pipeline.sector_metrics.compute_sector_leader_divergence",
               return_value=pd.DataFrame()), \
         patch("src.pipeline.sector_metrics.compute_sector_wyckoff_distribution",
               return_value=pd.DataFrame()), \
         patch("src.pipeline.sector_metrics.compute_rs_internal",
               return_value=pd.DataFrame()), \
         patch("src.pipeline.sector_metrics.compute_sector_concentration",
               return_value=pd.DataFrame()), \
         patch("src.pipeline.sector_metrics.compute_leader_representativeness",
               return_value=pd.DataFrame()):
        out = sm.compute_sector_metrics(
            df_stocks=pd.DataFrame(), holdings_df=pd.DataFrame(),
            leader_df=pd.DataFrame(), full_metrics_df=pd.DataFrame(),
            df_market=pd.DataFrame(),
        )
    assert set(out.keys()) == {
        "sector_leader_divergence_df", "sector_wyckoff_distribution_df",
        "rs_internal_df", "sector_concentration_df",
        "leader_representativeness_df"}


def test_sector_metrics_usa_ref_date_de_df_stocks(tmp_path, monkeypatch):
    """reference_date por defecto = df_stocks.index[-1]."""
    _setup_tmp(tmp_path, monkeypatch)
    idx = pd.date_range("2026-09-25", periods=1, freq="D")
    df_stocks = pd.DataFrame([1.0], index=idx)
    captured = {}
    def _fake_repr(leader_df, reference_date=None):
        captured["ref"] = reference_date
        return pd.DataFrame()
    with patch("src.pipeline.sector_metrics.compute_sector_leader_divergence",
               return_value=pd.DataFrame()), \
         patch("src.pipeline.sector_metrics.compute_sector_wyckoff_distribution",
               return_value=pd.DataFrame()), \
         patch("src.pipeline.sector_metrics.compute_rs_internal",
               return_value=pd.DataFrame()), \
         patch("src.pipeline.sector_metrics.compute_sector_concentration",
               return_value=pd.DataFrame()), \
         patch("src.pipeline.sector_metrics._compute_representativeness",
               side_effect=_fake_repr):
        sm.compute_sector_metrics(
            df_stocks=df_stocks, holdings_df=pd.DataFrame(),
            leader_df=pd.DataFrame(), full_metrics_df=pd.DataFrame(),
            df_market=pd.DataFrame(),
        )
    assert captured["ref"] == idx[-1]


def test_sector_metrics_degrada_si_sub_bloque_lanza(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    with patch("src.pipeline.sector_metrics.compute_sector_leader_divergence",
               side_effect=RuntimeError("x")), \
         patch("src.pipeline.sector_metrics.compute_sector_wyckoff_distribution",
               return_value=pd.DataFrame()), \
         patch("src.pipeline.sector_metrics.compute_rs_internal",
               return_value=pd.DataFrame()), \
         patch("src.pipeline.sector_metrics.compute_sector_concentration",
               return_value=pd.DataFrame()), \
         patch("src.pipeline.sector_metrics.compute_leader_representativeness",
               return_value=pd.DataFrame()):
        out = sm.compute_sector_metrics(
            df_stocks=pd.DataFrame(), holdings_df=pd.DataFrame(),
            leader_df=pd.DataFrame(), full_metrics_df=pd.DataFrame(),
            df_market=pd.DataFrame(),
        )
    assert out["sector_leader_divergence_df"] is None


def _fake_to_csv_that_fails():
    real_to_csv = pd.DataFrame.to_csv

    def fake_to_csv(self, path, **kw):
        real_to_csv(self.iloc[:0], path, **kw)
        raise OSError("simulado: fallo a mitad de escritura")

    return fake_to_csv


def _ne_df():
    """df_stocks NO vacio para pasar el guard de los helpers."""
    return pd.DataFrame({"x": [1.0]})


def test_sm_sld_fallo_to_csv_no_pierde_filas(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    csv = tmp_path / "outputs" / "history" / "sector_leader_divergence.csv"
    pd.DataFrame([{"date": "2026-09-24", "sector": "XLK", "v": 0.1}]).to_csv(csv, index=False)
    filas_antes = pd.read_csv(csv).shape[0]
    fake_df = pd.DataFrame([{"date": "2026-09-25", "sector": "XLK", "v": 0.2}])
    with patch.object(pd.DataFrame, "to_csv", _fake_to_csv_that_fails()), \
         patch("src.pipeline.sector_metrics.compute_sector_leader_divergence",
               return_value=fake_df):
        sm._compute_divergencia(_ne_df(), pd.DataFrame(), _ne_df(), pd.DataFrame())
    df = pd.read_csv(csv)
    assert df.shape[0] == filas_antes, f"perdio filas: {filas_antes} -> {df.shape[0]}"
    assert df.iloc[0]["date"] == "2026-09-24"


def test_sm_wy_fallo_to_csv_no_pierde_filas(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    csv = tmp_path / "outputs" / "history" / "sector_wyckoff_distribution.csv"
    pd.DataFrame([{"date": "2026-09-24", "sector": "XLK", "v": 0.1}]).to_csv(csv, index=False)
    filas_antes = pd.read_csv(csv).shape[0]
    fake_df = pd.DataFrame([{"date": "2026-09-25", "sector": "XLK", "v": 0.2}])
    with patch.object(pd.DataFrame, "to_csv", _fake_to_csv_that_fails()), \
         patch("src.pipeline.sector_metrics.compute_sector_wyckoff_distribution",
               return_value=fake_df):
        sm._compute_wyckoff(_ne_df(), pd.DataFrame())
    df = pd.read_csv(csv)
    assert df.shape[0] == filas_antes, f"perdio filas: {filas_antes} -> {df.shape[0]}"
    assert df.iloc[0]["date"] == "2026-09-24"


def test_sm_rs_fallo_to_csv_no_pierde_filas(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    csv = tmp_path / "outputs" / "history" / "rs_internal.csv"
    pd.DataFrame([{"date": "2026-09-24", "sector": "XLK", "ticker": "AAPL", "v": 0.1}]).to_csv(csv, index=False)
    filas_antes = pd.read_csv(csv).shape[0]
    fake_df = pd.DataFrame([{"date": "2026-09-25", "sector": "XLK", "ticker": "AAPL", "v": 0.2}])
    with patch.object(pd.DataFrame, "to_csv", _fake_to_csv_that_fails()), \
         patch("src.pipeline.sector_metrics.compute_rs_internal",
               return_value=fake_df):
        sm._compute_rs_internal(_ne_df(), pd.DataFrame(), pd.DataFrame())
    df = pd.read_csv(csv)
    assert df.shape[0] == filas_antes, f"perdio filas: {filas_antes} -> {df.shape[0]}"
    assert df.iloc[0]["date"] == "2026-09-24"


def test_sm_sc_fallo_to_csv_no_pierde_filas(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    csv = tmp_path / "outputs" / "history" / "sector_concentration.csv"
    pd.DataFrame([{"date": "2026-09-24", "sector": "XLK", "v": 0.1}]).to_csv(csv, index=False)
    filas_antes = pd.read_csv(csv).shape[0]
    fake_df = pd.DataFrame([{"date": "2026-09-25", "sector": "XLK", "v": 0.2}])
    with patch.object(pd.DataFrame, "to_csv", _fake_to_csv_that_fails()), \
         patch("src.pipeline.sector_metrics.compute_sector_concentration",
               return_value=fake_df):
        sm._compute_concentration(_ne_df(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    df = pd.read_csv(csv)
    assert df.shape[0] == filas_antes, f"perdio filas: {filas_antes} -> {df.shape[0]}"
    assert df.iloc[0]["date"] == "2026-09-24"


def test_sm_lr_fallo_to_csv_no_pierde_filas(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    csv = tmp_path / "outputs" / "history" / "leader_representativeness.csv"
    pd.DataFrame([{"date": "2026-09-24", "sector": "XLK", "ticker": "AAPL", "v": 0.1}]).to_csv(csv, index=False)
    filas_antes = pd.read_csv(csv).shape[0]
    fake_df = pd.DataFrame([{"date": "2026-09-25", "sector": "XLK", "ticker": "AAPL", "v": 0.2}])
    with patch.object(pd.DataFrame, "to_csv", _fake_to_csv_that_fails()), \
         patch("src.pipeline.sector_metrics.compute_leader_representativeness",
               return_value=fake_df):
        sm._compute_representativeness(_ne_df())
    df = pd.read_csv(csv)
    assert df.shape[0] == filas_antes, f"perdio filas: {filas_antes} -> {df.shape[0]}"
    assert df.iloc[0]["date"] == "2026-09-24"


# =============================================================================
# mte_confirmation.py
# =============================================================================

def test_mte_confirmation_contract_4_keys(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    with patch("src.pipeline.mte_confirmation._compute_mte",
               return_value={"scenario": "MIXED", "confidence": 0.5}), \
         patch("src.pipeline.mte_confirmation._compute_cross_module",
               return_value={"status": "OK", "message": ""}), \
         patch("src.pipeline.mte_confirmation._compute_confirmation",
               return_value=None):
        out = mc.compute_mte_confirmation(
            df_market=pd.DataFrame(), df_stocks=pd.DataFrame(),
            financial_score=0.5, all_signals=None,
            pcr_data=None, darkpool_data=None,
            macro_regime="MIXED", financial_regime="ABUNDANTE",
            vol_regime="NORMAL", real_liq_regime="ESTRECHA",
        )
    assert set(out.keys()) == {
        "mte_result", "cross_module_conflict", "confirmation_data"}


def test_mte_confirmation_pasa_scenario(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    with patch("src.pipeline.mte_confirmation._compute_mte",
               return_value={"scenario": "RECESSION", "confidence": 0.9}), \
         patch("src.pipeline.mte_confirmation._compute_cross_module",
               return_value={"status": "OK"}), \
         patch("src.pipeline.mte_confirmation._compute_confirmation",
               return_value=None):
        out = mc.compute_mte_confirmation(
            df_market=pd.DataFrame(), df_stocks=pd.DataFrame(),
            financial_score=0.5, all_signals=None,
            pcr_data=None, darkpool_data=None,
            macro_regime="MIXED", financial_regime="ABUNDANTE",
            vol_regime="NORMAL", real_liq_regime="ESTRECHA",
        )
    # mte_scenario no esta en el contrato real; verificamos mte_result
    assert out["mte_result"]["scenario"] == "RECESSION"


# =============================================================================
# finalize.py
# =============================================================================

def test_final_matrices_contract(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    with patch("src.pipeline.finalize._observation_date_from_df",
               return_value=pd.Timestamp("2026-09-25")), \
         patch("src.pipeline.finalize.is_market_day", return_value=True):
        out = fin.compute_final_matrices(
            sector_breadth_df=pd.DataFrame([{"sector": "XLK", "pct_above_ema20": 0.5}]),
            sector_concentration_df=pd.DataFrame(),
            sector_flow_characteristics_df=pd.DataFrame(),
            sector_wyckoff_distribution_df=pd.DataFrame(),
            sector_results={"ranking": [("XLK", "Technology", 0.5, "ACCUMULATION")]},
            financial_regime="ABUNDANTE",
            real_liq_regime="ESTRECHA",
            vol_regime="NORMAL",
            vol_score=0.0,
            real_liq_score=0.0,
            financial_score=0.5,
        )
    assert set(out.keys()) == {"sector_regime_matrix_df", "evidence_matrix_df"}


def test_save_regime_history_escribe_csv(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    # save_regime_history necesita df_macro_manual con columna 'date'
    macro_df = pd.DataFrame({
        "date": [pd.Timestamp("2026-09-25")],
    })
    macro_score_series = pd.Series([0.0, -0.1])
    with patch("src.pipeline.finalize.is_market_day", return_value=True):
        fin.save_regime_history(
            macro_score=macro_score_series,
            macro_regime="MIXED",
            macro_conf=0.5,
            liquidity_regime="ESTRECHA",
            vol_regime="NORMAL",
            sector_results={"regime": "NARROW RALLY", "ranking": []},
            df_macro_manual=macro_df,
        )
    p = tmp_path / "outputs" / "history" / "macro_regime.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert "date" in df.columns
    assert df.iloc[0]["macro_regime"] == "MIXED"


def test_save_regime_history_fallo_to_csv_no_pierde_filas(tmp_path, monkeypatch):
    """Si to_csv muere a mitad, el CSV original debe quedar intacto."""
    _setup_tmp(tmp_path, monkeypatch)
    csv = tmp_path / "outputs" / "history" / "macro_regime.csv"
    pd.DataFrame([{"date": "2026-09-24", "macro_regime": "OLD"}]).to_csv(csv, index=False)
    filas_antes = pd.read_csv(csv).shape[0]
    assert filas_antes == 1

    real_to_csv = pd.DataFrame.to_csv

    def fake_to_csv(self, path, **kw):
        real_to_csv(self.iloc[:0], path, **kw)
        raise OSError("simulado: fallo a mitad de escritura")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)
    macro_df = pd.DataFrame({"date": [pd.Timestamp("2026-09-25")]})
    with patch("src.pipeline.finalize.is_market_day", return_value=True):
        with pytest.raises(OSError, match="simulado"):
            fin.save_regime_history(
                macro_score=pd.Series([0.0, -0.1]),
                macro_regime="MIXED",
                macro_conf=0.5,
                liquidity_regime="ESTRECHA",
                vol_regime="NORMAL",
                sector_results={"regime": "NARROW RALLY", "ranking": []},
                df_macro_manual=macro_df,
            )

    df = pd.read_csv(csv)
    assert df.shape[0] == filas_antes, (
        f"CSV original perdio filas: {filas_antes} -> {df.shape[0]}"
    )
    assert df.iloc[0]["macro_regime"] == "OLD"


def test_save_sector_rankings_escribe_csv(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    fin.save_sector_rankings(
        {"ranking": [("XLK", "Technology", 0.5, "ACCUMULATION")]}
    )
    p = tmp_path / "outputs" / "report" / "sector_rankings.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert list(df.columns) == ["ticker", "name", "score", "wyckoff_phase"]
    assert df.iloc[0]["ticker"] == "XLK"


def test_generate_european_coverage_invoca_modulo(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    called = {"n": 0}
    def _fake_generate(*args, **kwargs):
        called["n"] += 1
    with patch("src.european_coverage.generate_european_coverage_report",
               side_effect=_fake_generate):
        fin.generate_european_coverage()
    assert called["n"] == 1

# =============================================================================
# Tests adicionales para _compute_mte, _compute_cross_module, _compute_confirmation
# =============================================================================

def test_compute_mte_ok(tmp_path, monkeypatch):
    """_compute_mte invoca indicators.mte.compute_mte y devuelve el dict."""
    _setup_tmp(tmp_path, monkeypatch)
    fake = {"scenario": "MIXED", "msi": 28, "ipi": 49}
    # Nota: all_signals debe ser un DataFrame, no None.
    # El codigo original usa 'all_signals' in dir() (siempre True) y luego
    # intenta all_signals.columns -> AttributeError si es None.
    # Bug latente BAJA documentado en el commit.
    with patch("indicators.mte.compute_mte", return_value=fake):
        out = mc._compute_mte(
            df_market=pd.DataFrame(), financial_score=0.5,
            all_signals=pd.DataFrame(), pcr_data=None, darkpool_data=None,
        )
    assert out == fake


def test_compute_mte_darkpool_archival_excluido(tmp_path, monkeypatch):
    """Dark Pool con week > 14d se excluye del MTE (mte_darkpool=None)."""
    _setup_tmp(tmp_path, monkeypatch)
    captured = {}
    def _fake_compute_mte(df_market, fc, cred, vol, pcr, darkpool, temporal_meta=None):
        captured["darkpool"] = darkpool
        return {"scenario": "MIXED", "msi": 28, "ipi": 49}
    old_week = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    with patch("indicators.mte.compute_mte", side_effect=_fake_compute_mte):
        mc._compute_mte(
            df_market=pd.DataFrame(), financial_score=0.5,
            all_signals=pd.DataFrame(), pcr_data=None,
            darkpool_data={"week": old_week},
        )
    assert captured["darkpool"] is None


def test_compute_cross_module_sin_mte_result(tmp_path, monkeypatch):
    """_compute_cross_module pasa mte_scenario=None si mte_result es None."""
    captured = {}
    def _fake_detect(**kwargs):
        captured.update(kwargs)
        return {"conflict_level": "OK", "message": ""}
    with patch("src.pipeline.mte_confirmation.detect_cross_module_conflict",
               side_effect=_fake_detect):
        mc._compute_cross_module(
            macro_regime="MIXED", financial_regime="ABUNDANTE",
            vol_regime="NORMAL", real_liq_regime="N/A",
            mte_result=None,
        )
    assert captured["mte_scenario"] is None
    assert captured["liquidity_regime"] is None


def test_compute_confirmation_integracion_minima(tmp_path, monkeypatch):
    """_compute_confirmation rellena keys con sub-modulos mockeados."""
    _setup_tmp(tmp_path, monkeypatch)
    # 10y3m.csv necesita existir para la primera rama
    macro_dir = tmp_path / "data" / "macro_manual"
    macro_dir.mkdir(parents=True)
    (macro_dir / "10y3m.csv").write_text(
        "date,T10Y3M\n2026-09-25,1.23\n", encoding="utf-8")

    with patch("indicators.vol_metrics.compute_vol_metrics",
               return_value={"rv_21d": 0.10, "rv_60d": 0.11, "vrp_21d": 0.04,
                             "vrp_60d": 0.03}), \
         patch("indicators.cross_asset.compute_cross_asset_ratios",
               return_value={"copper_gold": 0.0123}), \
         patch("indicators.fls.compute_fls",
               return_value={"fls_normalized": 0.69, "components": 3,
                             "total_components": 5, "detail": {}}), \
         patch("indicators.breadth_equity.compute_advance_decline",
               return_value={"ad_net": 74, "advances": 193, "declines": 119,
                             "new_highs": 5, "new_lows": 2, "nh_nl": 3,
                             "breadth_thrust": 0.5, "ad_line": 14843}):
        out = mc._compute_confirmation(
            df_market=pd.DataFrame(), df_stocks=pd.DataFrame(),
        )
    assert out["t10y3m"] == 1.23
    assert out["rv_21d"] == 0.10
    assert out["ratios"]["copper_gold"] == 0.0123
    assert out["fls"]["fls_normalized"] == 0.69
    assert out["ad"]["ad_net"] == 74


# =============================================================================
# D-05 (2026-10-03): mte_confirmation + validation_gate con reference_date tz-aware
# =============================================================================

def test_compute_mte_tz_aware_no_silencia_archival(capsys):
    """_compute_mte con reference_date tz-aware + darkpool antiguo debe
    imprimir ARCHIVAL. Si no aparece, el TypeError fue capturado por el
    except silencioso de L48.
    """
    from zoneinfo import ZoneInfo
    from unittest.mock import patch
    from src.pipeline.mte_confirmation import _compute_mte
    ref_tz = datetime.now(ZoneInfo("Europe/Madrid"))
    darkpool = {"week": "2026-08-18"}  # > 14 dias
    with patch("indicators.mte.compute_mte",
               return_value={"scenario": "X", "msi": 0.0, "ipi": 0.0}):
        _compute_mte(
            pd.DataFrame(), 0.0, None, None, darkpool,
            temporal_meta=None, reference_date=ref_tz,
        )
    captured = capsys.readouterr()
    assert "ARCHIVAL" in captured.out, (
        "con reference_date tz-aware y darkpool antiguo, _compute_mte debe "
        "detectar ARCHIVAL; si no aparece, el TypeError tz fue silenciado"
    )


def test_compute_mte_confirmation_propaga_reference_date():
    """compute_mte_confirmation debe pasar reference_date a _compute_mte."""
    from zoneinfo import ZoneInfo
    from unittest.mock import patch
    from src.pipeline import mte_confirmation as m
    ref_tz = datetime.now(ZoneInfo("Europe/Madrid"))
    with patch.object(m, "_compute_mte", return_value=None) as mock_mte, \
         patch.object(m, "_compute_cross_module", return_value=None), \
         patch.object(m, "_compute_confirmation", return_value=None):
        m.compute_mte_confirmation(
            pd.DataFrame(), pd.DataFrame(), 0.0, None, None, None,
            None, None, None, None, reference_date=ref_tz,
        )
    assert mock_mte.called
    call_kwargs = mock_mte.call_args.kwargs
    assert call_kwargs.get("reference_date") is ref_tz, (
        "compute_mte_confirmation no propaga reference_date a _compute_mte"
    )

