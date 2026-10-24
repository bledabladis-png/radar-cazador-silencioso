"""Tests de compute_wyckoff_holdings_report.

Sigue el patron de test_sector_wyckoff_distribution.py: fixtures
sinteticos MultiIndex + monkeypatch sobre classify_phase.
"""
import pandas as pd
import pytest

from indicators import wyckoff_holdings_report as whr


def _make_df_stocks(tickers, n_rows=500, start="2024-01-01"):
    dates = pd.date_range(start, periods=n_rows, freq="B")
    data = {}
    for field in ["Open", "High", "Low", "Close", "Volume"]:
        for t in tickers:
            data[(field, t)] = 100.0
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _make_sectors_holdings(etf, tickers, weights=None):
    if weights is None:
        weights = [100.0 / len(tickers)] * len(tickers)
    return pd.DataFrame({
        "etf": [etf] * len(tickers),
        "ticker": list(tickers),
        "weight": weights,
    })


def _make_indices_holdings(etf, tickers, weights=None):
    if weights is None:
        weights = [100.0 / len(tickers)] * len(tickers)
    return pd.DataFrame({
        "etf": [etf] * len(tickers),
        "ticker": list(tickers),
        "name": ["{} Inc".format(t) for t in tickers],
        "weight": weights,
    })


class _FakeFP:
    def __init__(self, phase, event_phase=None, phase_since=None, basis=""):
        self.phase = phase
        self.event_phase = event_phase
        self.phase_since = phase_since
        self.current_phase_basis = basis


def _patch_classifier(monkeypatch, mapping):
    def fake_classify(df, ticker):
        ph = mapping.get(ticker)
        if ph is None:
            return _FakeFP("INSUFFICIENT_DATA")
        return _FakeFP(ph, basis="{} {}".format(ticker, ph))
    monkeypatch.setattr(whr, "classify_phase", fake_classify)


def test_empty_inputs_return_empty_tables():
    df = _make_df_stocks(["A"])
    out = whr.compute_wyckoff_holdings_report(
        df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )
    assert set(out.keys()) == {
        "usa_sector_acc", "usa_sector_dist",
        "usa_index_acc", "usa_index_dist",
        "eu_index_acc", "eu_index_dist",
    }
    for key in out:
        assert out[key].empty
        assert list(out[key].columns) == whr.OUTPUT_COLUMNS


def test_sector_acc_and_dist_split(monkeypatch):
    tickers = ["AAPL", "MSFT", "NVDA", "XYZ", "ABC"]
    df = _make_df_stocks(tickers)
    sectors = _make_sectors_holdings("XLK", tickers, weights=[30, 25, 20, 15, 10])
    _patch_classifier(monkeypatch, {
        "AAPL": "ACCUMULATION",
        "MSFT": "MARKUP",
        "NVDA": "RANGE",
        "XYZ": "DISTRIBUTION_FORMING",
        "ABC": "MARKDOWN",
    })
    out = whr.compute_wyckoff_holdings_report(
        df, sectors, pd.DataFrame(), pd.DataFrame()
    )
    acc = out["usa_sector_acc"]
    dist = out["usa_sector_dist"]
    assert set(acc["ticker"]) == {"AAPL", "MSFT"}
    assert set(dist["ticker"]) == {"XYZ", "ABC"}
    assert "NVDA" not in acc["ticker"].tolist()
    assert "NVDA" not in dist["ticker"].tolist()


def test_ordering_phase_then_weight(monkeypatch):
    tickers = ["T1", "T2", "T3"]
    df = _make_df_stocks(tickers)
    sectors = _make_sectors_holdings("XLK", tickers, weights=[50, 30, 20])
    _patch_classifier(monkeypatch, {
        "T1": "MARKUP",
        "T2": "ACCUMULATION",
        "T3": "ACCUMULATION_FORMING",
    })
    out = whr.compute_wyckoff_holdings_report(
        df, sectors, pd.DataFrame(), pd.DataFrame()
    )
    acc = out["usa_sector_acc"]
    assert list(acc["ticker"]) == ["T2", "T3", "T1"]


def test_ordering_weight_desc_within_same_phase(monkeypatch):
    tickers = ["T1", "T2", "T3"]
    df = _make_df_stocks(tickers)
    sectors = _make_sectors_holdings("XLK", tickers, weights=[50, 30, 20])
    _patch_classifier(monkeypatch, {
        "T1": "MARKUP",
        "T2": "MARKUP",
        "T3": "ACCUMULATION",
    })
    out = whr.compute_wyckoff_holdings_report(
        df, sectors, pd.DataFrame(), pd.DataFrame()
    )
    acc = out["usa_sector_acc"]
    assert list(acc["ticker"]) == ["T3", "T1", "T2"]


def test_ticker_below_200_bars_excluded(monkeypatch):
    df = _make_df_stocks(["SHORT"]).iloc[:100]
    sectors = _make_sectors_holdings("XLK", ["SHORT"], weights=[100])
    _patch_classifier(monkeypatch, {"SHORT": "ACCUMULATION"})
    out = whr.compute_wyckoff_holdings_report(
        df, sectors, pd.DataFrame(), pd.DataFrame()
    )
    assert out["usa_sector_acc"].empty


def test_insufficient_data_excluded_from_both(monkeypatch):
    df = _make_df_stocks(["T1"])
    sectors = _make_sectors_holdings("XLK", ["T1"], weights=[100])
    _patch_classifier(monkeypatch, {"T1": None})
    out = whr.compute_wyckoff_holdings_report(
        df, sectors, pd.DataFrame(), pd.DataFrame()
    )
    assert out["usa_sector_acc"].empty
    assert out["usa_sector_dist"].empty


def test_warm_up_incomplete_marked_in_basis(monkeypatch):
    df = _make_df_stocks(["WARM"]).iloc[:300]
    sectors = _make_sectors_holdings("XLK", ["WARM"], weights=[100])
    _patch_classifier(monkeypatch, {"WARM": "ACCUMULATION"})
    out = whr.compute_wyckoff_holdings_report(
        df, sectors, pd.DataFrame(), pd.DataFrame()
    )
    acc = out["usa_sector_acc"]
    assert len(acc) == 1
    assert "warm_up_incompleto" in acc.iloc[0]["basis"]
    assert "n=300" in acc.iloc[0]["basis"]


def test_coverage_weight_pct_and_top20_limit(monkeypatch):
    tickers = ["T{:02d}".format(i) for i in range(25)]
    weights = [float(w) for w in range(25, 0, -1)]
    df = _make_df_stocks(tickers)
    sectors = _make_sectors_holdings("XLK", tickers, weights=weights)
    _patch_classifier(monkeypatch, {t: "ACCUMULATION" for t in tickers})
    out = whr.compute_wyckoff_holdings_report(
        df, sectors, pd.DataFrame(), pd.DataFrame()
    )
    acc = out["usa_sector_acc"]
    assert len(acc) == 20
    expected_cov = float(sum(range(6, 26)))
    assert acc.iloc[0]["coverage_weight_pct"] == pytest.approx(expected_cov)
    assert "T24" not in acc["ticker"].tolist()
    assert "T20" not in acc["ticker"].tolist()


def test_index_uses_name_column(monkeypatch):
    tickers = ["AAPL", "MSFT"]
    df = _make_df_stocks(tickers)
    idx = _make_indices_holdings("SPY", tickers, weights=[50, 50])
    _patch_classifier(monkeypatch, {"AAPL": "ACCUMULATION", "MSFT": "MARKUP"})
    out = whr.compute_wyckoff_holdings_report(
        df, pd.DataFrame(), idx, pd.DataFrame()
    )
    acc = out["usa_index_acc"]
    assert set(acc["name"]) == {"AAPL Inc", "MSFT Inc"}
    assert set(acc["group_type"]) == {"index_usa"}


def test_eu_index_type_and_ordering(monkeypatch):
    tickers = ["ASML.AS", "SAP.DE"]
    df = _make_df_stocks(tickers)
    eu = _make_indices_holdings("DAXEX", tickers, weights=[50, 50])
    _patch_classifier(monkeypatch, {
        "ASML.AS": "DISTRIBUTION",
        "SAP.DE": "DISTRIBUTION_FORMING",
    })
    out = whr.compute_wyckoff_holdings_report(
        df, pd.DataFrame(), pd.DataFrame(), eu
    )
    dist = out["eu_index_dist"]
    assert set(dist["group_type"]) == {"index_eu"}
    assert list(dist["ticker"]) == ["ASML.AS", "SAP.DE"]