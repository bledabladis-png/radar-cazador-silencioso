"""Tests pipeline: diagnostics (fase 9a) + engines (fase 8a).

Cobertura de orquestadores. Los modulos subyacentes ya estan cubiertos.
Aqui se verifica contrato de retorno, manejo de degradacion (df vacio ->
fallback), logica de forzar lideres SLPM, y persistencia sectorial.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.diagnostics import (
    compute_diagnostics, _compute_directional_agreement,
    _compute_price_flow_divergence, _compute_shock_sensitivity,
    SECTOR_ETFS,
)
from src.pipeline.engines import (
    compute_engines, _forzar_lideres_slpm,
    _compute_tactical_structural, _compute_persistence_and_save,
)


# =============================================================================
# diagnostics.py
# =============================================================================

def test_diagnostics_contract_4_keys():
    """compute_diagnostics devuelve las 4 keys documentadas incluso si
    los modulos subyacentes degradan."""
    with patch("src.pipeline.diagnostics.compute_signal_agreement",
               return_value={"agreement": 0.5, "display": "50% MIXED"}), \
         patch("src.pipeline.diagnostics.detect_price_flow_divergence",
               return_value={"status": "ALIGNED", "message": ""}), \
         patch("indicators.commodity_market_correlation.compute_commodity_market_correlation",
               return_value={}):
        out = compute_diagnostics(
            df_market=pd.DataFrame(),
            tactical_scores={s: 0.1 for s in SECTOR_ETFS},
            structural_scores={s: 0.1 for s in SECTOR_ETFS},
            sector_flow_rank=[("XLK", 0.2)],
        )
    assert set(out.keys()) == {
        "signal_agreements", "signal_agreements_display",
        "price_flow_divergences", "shock_sensitivities"}


def test_directional_agreement_degradacion_sin_df():
    """Sin df_market util, get_col lanza y se aplica el fallback de 11 sectores."""
    with patch("src.pipeline.diagnostics.get_col", side_effect=KeyError("missing")):
        ag, disp = _compute_directional_agreement(
            pd.DataFrame(),
            tactical_scores={},
            structural_scores={},
            sector_flow_rank=[],
        )
    assert set(ag.keys()) == set(SECTOR_ETFS)
    assert set(disp.keys()) == set(SECTOR_ETFS)


def test_price_flow_divergence_devuelve_11_sectores():
    with patch("src.pipeline.diagnostics.get_col", side_effect=KeyError("x")), \
         patch("src.pipeline.diagnostics.detect_price_flow_divergence",
               return_value={"status": "ALIGNED", "message": ""}):
        out = _compute_price_flow_divergence(pd.DataFrame(), sector_flow_rank=[])
    assert set(out.keys()) == set(SECTOR_ETFS)


def test_shock_sensitivity_degradacion_sin_df():
    with patch("indicators.commodity_market_correlation.compute_commodity_market_correlation",
               side_effect=RuntimeError("x")):
        out = _compute_shock_sensitivity(pd.DataFrame())
    assert set(out.keys()) == set(SECTOR_ETFS)
    for v in out.values():
        assert v == {}


# =============================================================================
# engines.py
# =============================================================================

def test_engines_contract_5_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    with patch("src.pipeline.engines._compute_tactical_structural",
               return_value=({}, {})), \
         patch("src.pipeline.engines._compute_persistence_and_save",
               return_value={}):
        out = compute_engines(
            df_market=pd.DataFrame(),
            sector_results={"ranking": [("XLK", "Technology", 0.5, "")]},
            sector_flow_rank=[("XLK", 0.2)],
            otros_flow_rank=[],
            leader_df=None,
        )
    assert set(out.keys()) == {
        "leader_metrics_for_slpm", "top_sector_flow",
        "tactical_scores", "structural_scores", "sector_persistence"}


def test_forzar_lideres_slpm_sin_leader_df():
    metrics, flow = _forzar_lideres_slpm(
        sector_results={"ranking": [("XLK", "Technology", 0.5, "")]},
        sector_flow_rank=[("XLK", 0.2)],
        otros_flow_rank=[],
        leader_df=None,
    )
    assert metrics == []
    assert flow == 0.2


def test_forzar_lideres_slpm_extrae_5_tickers():
    leader_df = pd.DataFrame([
        {"sector": "XLK", "ticker": f"T{i}", "rs": 1.0, "rs_mom": 0.1,
         "flow_proxy_z": 0.3, "wyckoff_phase": "MARKUP"}
        for i in range(7)
    ])
    metrics, flow = _forzar_lideres_slpm(
        sector_results={"ranking": [("XLK", "Technology", 0.5, "")]},
        sector_flow_rank=[("XLK", 0.5)],
        otros_flow_rank=[],
        leader_df=leader_df,
    )
    assert len(metrics) == 5
    for m in metrics:
        assert set(m.keys()) == {"ticker", "rs", "rs_momentum", "flow_proxy_z", "wyckoff_phase"}


def test_tactical_structural_degradacion_sin_df():
    """Si los engines subyacentes fallan, cada sector queda a 0.0."""
    with patch("regimes.tactical_engine.compute_tactical_score",
               side_effect=RuntimeError("x")), \
         patch("regimes.structural_engine.compute_structural_score",
               side_effect=RuntimeError("x")):
        t, s = _compute_tactical_structural(pd.DataFrame())
    assert set(t.keys()) == set(SECTOR_ETFS)
    assert set(s.keys()) == set(SECTOR_ETFS)
    for v in t.values():
        assert v == 0.0
    for v in s.values():
        assert v == 0.0


def test_persistence_degradacion_sin_df(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    with patch("src.pipeline.engines.get_col", side_effect=KeyError("x")):
        out = _compute_persistence_and_save(pd.DataFrame())
    assert set(out.keys()) == set(SECTOR_ETFS)
    for v in out.values():
        assert v is None
    # El CSV debe haberse escrito con los 11 sectores
    p = tmp_path / "outputs" / "history" / "sector_persistence.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert set(df["sector"].unique()) == set(SECTOR_ETFS)


def test_persistence_fallo_to_csv_no_pierde_filas(tmp_path, monkeypatch):
    """Si to_csv muere a mitad, el CSV original debe quedar intacto."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True)
    csv = tmp_path / "outputs" / "history" / "sector_persistence.csv"
    pd.DataFrame([{"date": "2026-09-24", "sector": "XLK", "persistence": 0.5}]).to_csv(csv, index=False)
    filas_antes = pd.read_csv(csv).shape[0]
    assert filas_antes == 1

    real_to_csv = pd.DataFrame.to_csv

    def fake_to_csv(self, path, **kw):
        real_to_csv(self.iloc[:0], path, **kw)
        raise OSError("simulado: fallo a mitad de escritura")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fake_to_csv)
    with patch("src.pipeline.engines.get_col", side_effect=KeyError("x")):
        _compute_persistence_and_save(pd.DataFrame())

    df = pd.read_csv(csv)
    assert df.shape[0] == filas_antes, (
        f"CSV original perdio filas: {filas_antes} -> {df.shape[0]}"
    )
    assert df.iloc[0]["date"] == "2026-09-24"