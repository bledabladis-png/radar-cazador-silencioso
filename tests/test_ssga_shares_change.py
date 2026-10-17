# -*- coding: utf-8 -*-
"""Regresion H3 (2026-10-17): shares_change no debe ser NaN por diff
sobre 1 fila.

Antes: _compute_primary_flow no anadia shares_change al df. last_df
(1 fila por ticker tras groupby.tail(1)) derivaba shares_change con
diff() -> NaN. El reporte mostraba N/D en dShares aunque el dato
existia en la serie completa.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.providers.ssga_fund_data import _compute_primary_flow


def _make_df(shares):
    dates = pd.date_range("2026-10-11", periods=len(shares), freq="D")
    return pd.DataFrame({
        "Date": dates,
        "nav": [65.0] * len(shares),
        "shares_outstanding": shares,
        "total_net_assets": [4.1e9] * len(shares),
    })


def test_shares_change_presente_tras_compute():
    df = _compute_primary_flow(_make_df([100, 100, 100]))
    assert "shares_change" in df.columns


def test_shares_change_es_diff_real():
    df = _compute_primary_flow(_make_df([100, 110, 105]))
    assert df["shares_change"].iloc[1] == 10
    assert df["shares_change"].iloc[2] == -5


def test_shares_change_cero_cuando_constant():
    """FEZ: shares constantes -> shares_change=0, no NaN."""
    df = _compute_primary_flow(_make_df([100, 100, 100]))
    assert df["shares_change"].iloc[1] == 0
    assert df["shares_change"].iloc[2] == 0