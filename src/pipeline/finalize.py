# -*- coding: utf-8 -*-
"""Fase 12 del pipeline: matrices finales + side effects + cobertura europea.

Extraido de run.py (refactor C2, fase C2-12). Incluye:
- compute_final_matrices (Matriz Regimen + Matriz Evidencia)
- save_regime_history (movido de report_generator.py via C1-10)
- save_sector_rankings (movido de report_generator.py via C1-10)
- generate_european_coverage
"""

import os
from pathlib import Path

import pandas as pd

from src.utils import _observation_date_from_df
from src.market_calendar import is_market_day


def compute_final_matrices(sector_breadth_df, sector_concentration_df,
                           sector_flow_characteristics_df,
                           sector_wyckoff_distribution_df, sector_results,
                           financial_regime, real_liq_regime,
                           vol_regime, vol_score, real_liq_score, financial_score):
    """Construye Matriz de Regimen Sectorial + Matriz de Evidencia.

    Returns:
        dict con keys: sector_regime_matrix_df, evidence_matrix_df
    """
    sector_regime_matrix_df = None
    try:
        from indicators.sector_regime_matrix import build_sector_regime_matrix
        sector_regime_matrix_df = build_sector_regime_matrix(
            sector_breadth_df, sector_flow_characteristics_df, sector_results
        )
        if sector_regime_matrix_df is not None and not sector_regime_matrix_df.empty:
            mp_path = Path('outputs/history/sector_regime_matrix.csv')
            mp_path.parent.mkdir(parents=True, exist_ok=True)
            sector_regime_matrix_df.to_csv(mp_path, index=False, encoding='utf-8')
            print("  Matriz de regimen sectorial calculada.")
        else:
            sector_regime_matrix_df = None
    except Exception as e:
        print(f"  Matriz de regimen sectorial omitida: {e}")
        sector_regime_matrix_df = None

    evidence_matrix_df = None
    try:
        from indicators.evidence_matrix import compute_evidence_matrix
        evidence_matrix_df = compute_evidence_matrix(
            sector_breadth_df,
            sector_concentration_df,
            sector_flow_characteristics_df,
            sector_wyckoff_distribution_df,
            liquidity_regime=financial_regime,
            real_liquidity_regime=real_liq_regime,
            volatility_regime=vol_regime,
            volatility_score=vol_score,
            liquidity_score=real_liq_score if real_liq_score is not None else financial_score
        )
        if evidence_matrix_df is not None and not evidence_matrix_df.empty:
            em_path = Path('outputs/history/evidence_matrix.csv')
            em_path.parent.mkdir(parents=True, exist_ok=True)
            evidence_matrix_df.to_csv(em_path, index=False)
            print("  Matriz de evidencia calculada.")
        else:
            evidence_matrix_df = None
    except Exception as e:
        print(f"  Matriz de evidencia omitida: {e}")
        evidence_matrix_df = None

    return {
        'sector_regime_matrix_df': sector_regime_matrix_df,
        'evidence_matrix_df': evidence_matrix_df,
    }


def save_regime_history(macro_score, macro_regime, macro_conf,
                        liquidity_regime, vol_regime, sector_results,
                        df_macro_manual=None):
    """Persiste la fila del regimen actual en outputs/history/macro_regime.csv.

    Movido de report_generator.py (C1-10) y consolidado aqui (C2-12).
    """
    hist_path = "outputs/history/macro_regime.csv"
    obs_date = _observation_date_from_df(df_macro_manual, col='date')
    if obs_date is None:
        print("  [WARN] save_regime_history: sin fecha macro valida. Se omite escritura.")
        return
    # B5-followup (2026-09-12): segundo candado. El bot FRED publica iorb.csv con
    # fecha del dia natural (incluye sabado/domingo). Si obs_date no es sesion NYSE,
    # se omite la escritura para no contaminar el historico con filas B2.
    obs_date_only = pd.Timestamp(obs_date).date()
    if not is_market_day(obs_date_only):
        print(f"  [WARN] save_regime_history: obs_date {obs_date_only} no es sesion NYSE. Se omite escritura.")
        return
    new_row = pd.DataFrame({
        "date": [pd.Timestamp(obs_date).strftime('%Y-%m-%d')],
        "macro_regime": [macro_regime],
        "macro_score": [macro_score.iloc[-1]],
        "macro_conf": [macro_conf],
        "liquidity_regime": [liquidity_regime],
        "volatility_regime": [vol_regime],
        "sector_regime": [sector_results["regime"]],
    })
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path, dtype=str)
        hist = pd.concat([hist, new_row.astype(str)], ignore_index=True)
        hist = hist.drop_duplicates(subset=['date'], keep='last')
    else:
        hist = new_row
    hist.to_csv(hist_path, index=False)


def save_sector_rankings(sector_results):
    """Persiste el ranking de sectores en outputs/report/sector_rankings.csv.

    Movido de report_generator.py (C1-10) y consolidado aqui (C2-12).
    """
    sector_df = pd.DataFrame(
        sector_results["ranking"],
        columns=["ticker", "name", "score", "wyckoff_phase"],
    )
    sector_df.to_csv("outputs/report/sector_rankings.csv", index=False)


def generate_european_coverage():
    """Genera reporte de cobertura europea (descriptivo; no rompe el run si falla)."""
    try:
        from src.european_coverage import generate_european_coverage_report
        generate_european_coverage_report()
    except Exception as e:
        print(f"  Cobertura europea omitida: {e}")
