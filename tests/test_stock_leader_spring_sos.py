# -*- coding: utf-8 -*-
"""Regresion H5 (2026-10-17): Spring/SOS se pintan como 0/1 explicitos.

Antes: '[v]' si 1, celda vacia si 0. El lector no podia distinguir
"spring=0" de "spring=NaN". 0 es un valor real (calculado por
compute_wls), no un hueco.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _build_fake_wls():
    return pd.DataFrame([
        {"ticker": "AAA", "sector": "XLK", "rs": 1.0, "rs_mom": 0.05,
         "flow_proxy_z": 0.5, "wyckoff_score": 0.6, "wyckoff_phase": "MARKUP",
         "persistence_5d": 0.8, "persistence_10d": 0.7, "persistence_20d": 0.6,
         "stability": 0.9, "spring": 0, "sos": 1, "wls": 1.2,
         "sector_rank_pct": 0.9},
    ])


def test_spring_sos_cero_se_muestran_como_cero(monkeypatch):
    """spring=0 aparece como '0' en la celda, no vacio."""
    from indicators import stock_leader as sl

    monkeypatch.setattr(sl, "compute_stock_metrics",
                        lambda *a, **k: _build_fake_wls())
    monkeypatch.setattr(sl, "compute_wls", lambda df: _build_fake_wls())
    monkeypatch.setattr(sl, "TOP_N_CANDIDATES", 5)
    monkeypatch.setattr(sl, "TOP_N_LEADERS", 5)

    holdings = pd.DataFrame({"etf": ["XLK"], "ticker": ["AAA"], "weight": [1.0]})
    lines, _, _ = sl.generate_leader_section(
        df_market=pd.DataFrame(),
        df_stocks=pd.DataFrame(),
        holdings_df=holdings,
        fase_dict={"XLK": "MARKUP"},
        operabilidad_dict={"XLK": "OPORTUNIDAD MODERADA"},
    )
    text = "".join(lines)
    assert "| 0 | 1 |" in text, (
        f"Se esperaba '| 0 | 1 |' (spring=0, sos=1) en la fila. "
        f"Got: {text[:500]}"
    )
    assert "| [v] |" not in text
    assert "|  |  |" not in text