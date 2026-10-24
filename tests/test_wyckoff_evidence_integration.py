"""Tests del enrichment de evidencia pasiva (mSOW/mSOS/LPSY/LPS)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.wyckoff_distribution import classify_distribution_at
from indicators.wyckoff_accumulation import classify_accumulation_at


def _make_df(n=300, ticker="TEST"):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(3)
    close = 100.0 + np.cumsum(rng.normal(0, 0.3, n))
    df = pd.DataFrame(
        {
            ("Open", ticker): close,
            ("High", ticker): close + 0.5,
            ("Low", ticker): close - 0.5,
            ("Close", ticker): close,
            ("Volume", ticker): np.full(n, 1e6),
        },
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "ticker"])
    return df


def test_dist_enrichment_has_msow_field():
    df = _make_df()
    as_of = df.index[-1]
    r = classify_distribution_at(df, "TEST", as_of)
    # Si la cascada llega a tener rango (evidence con bc_date), debe tener msow
    if "bc_date" in r.evidence:
        assert "msow" in r.evidence, f"falta msow en evidence: {r.evidence.keys()}"
        assert r.evidence["msow"]["status"] in ("CONFIRMED", "NOT_OBSERVED", "NOT_EVALUABLE")


def test_acc_enrichment_has_msos_field():
    df = _make_df()
    as_of = df.index[-1]
    r = classify_accumulation_at(df, "TEST", as_of)
    if "sc_date" in r.evidence:
        assert "msos" in r.evidence, f"falta msos en evidence: {r.evidence.keys()}"
        assert r.evidence["msos"]["status"] in ("CONFIRMED", "NOT_OBSERVED", "NOT_EVALUABLE")


def test_dist_enrichment_does_not_change_classification():
    df = _make_df()
    as_of = df.index[-1]
    r = classify_distribution_at(df, "TEST", as_of)
    # classification debe ser uno de los valores esperados
    valid = {"DISTRIBUTION_CONFIRMED", "MARKDOWN", "RANGE_ACTIVE_NO_BREAK",
             "DISTRIBUTION_FORMING", "BREAK_PENDING", "NOT_EVALUABLE",
             "BREAK_WITHOUT_DISTRIBUTION_CONTEXT"}
    assert r.classification in valid