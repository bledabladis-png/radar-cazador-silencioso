# -*- coding: utf-8 -*-
"""Test: _fetch_liquidity_v2_data usa CSVs cuando existen.

Item: REP-AUDIT-20261022-LIQ-V2-DATA-SOURCE
Causa: v2 leia SOFR/EFFR solo de FRED API (con lag). CSVs diarios
del bot suelen ir 1-2 dias por delante. State quedaba stale.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.pipeline.regimes import _fetch_liquidity_v2_data
from regimes.liquidity_v2 import _score_continuo


def test_score_continuo_zero_es_cero_positivo():
    """diff_bp=0 -> score 0.0, no -0.0."""
    s = _score_continuo(0.0)
    assert s == 0.0
    assert str(s) == "0.0", f"score es {s!r}, se serializaria como {str(s)}"


def test_fetch_usa_csvs_si_existen():
    """Con CSVs presentes, la fecha maxima devuelta debe coincidir con los CSVs."""
    sofr_p = Path("data/macro_manual/sofr.csv")
    effr_p = Path("data/macro_manual/effr.csv")
    if not sofr_p.exists() or not effr_p.exists():
        pytest.skip("CSVs sofr/effr no presentes en este entorno")

    sofr_csv = pd.read_csv(sofr_p, index_col=0, parse_dates=True)
    effr_csv = pd.read_csv(effr_p, index_col=0, parse_dates=True)
    common_csv = sofr_csv.index.intersection(effr_csv.index)
    expected_last = common_csv.max()

    sofr, effr, _ = _fetch_liquidity_v2_data()
    assert sofr is not None and effr is not None
    assert effr.index.max() == expected_last, (
        f"fetch devuelve {effr.index.max()}, CSV comun tiene {expected_last}"
    )


def test_fetch_fallback_api_si_no_hay_csvs(tmp_path, monkeypatch):
    """Sin CSVs, cae a API (mockeada)."""
    monkeypatch.chdir(tmp_path)
    import pandas_datareader.data as web
    idx = pd.date_range("2026-10-01", "2026-10-21", freq="B")
    df = pd.DataFrame({"FAKE": [3.88] * len(idx)}, index=idx)
    monkeypatch.setattr(web, "DataReader", lambda *a, **kw: df.copy())

    sofr, effr, _ = _fetch_liquidity_v2_data()
    assert sofr is not None and effr is not None
    assert sofr.index.max() == idx.max()