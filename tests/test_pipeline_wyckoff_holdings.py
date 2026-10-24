"""Tests de src.pipeline.wyckoff_holdings.

Mockea persist/skip y compute_wyckoff_holdings_report para verificar
el contrato del wrapper sin tocar CSV reales.
"""
import pandas as pd

from src.pipeline import wyckoff_holdings as wh


ALL_KEYS = [
    "usa_sector_acc", "usa_sector_dist",
    "usa_index_acc", "usa_index_dist",
    "eu_index_acc", "eu_index_dist",
]

ALL_NAMES = [
    "wyckoff_usa_sector_accumulation",
    "wyckoff_usa_sector_distribution",
    "wyckoff_usa_index_accumulation",
    "wyckoff_usa_index_distribution",
    "wyckoff_eu_index_accumulation",
    "wyckoff_eu_index_distribution",
]

OUTPUT_COLUMNS = [
    "date", "group", "group_type", "ticker", "name", "weight",
    "phase", "event_phase", "phase_since", "close", "basis",
    "coverage_weight_pct",
]


def _make_df_stocks(tickers, n_rows=500):
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    data = {}
    for field in ["Open", "High", "Low", "Close", "Volume"]:
        for t in tickers:
            data[(field, t)] = 100.0
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _make_holdings():
    return pd.DataFrame({
        "etf": ["XLK", "XLK", "XLE"],
        "ticker": ["AAPL", "MSFT", "XOM"],
        "weight": [50.0, 50.0, 100.0],
    })


def _make_index_holdings():
    return pd.DataFrame({
        "etf": ["SPY", "SPY", "DAXEX", "DAXEX"],
        "ticker": ["AAPL", "MSFT", "SAP.DE", "ASML.AS"],
        "name": ["AAPL", "MSFT", "SAP", "ASML"],
        "weight": [50.0, 50.0, 50.0, 50.0],
    })


def _make_tables(empty_keys=()):
    out = {}
    for k in ALL_KEYS:
        if k in empty_keys:
            out[k] = pd.DataFrame(columns=OUTPUT_COLUMNS)
        else:
            out[k] = pd.DataFrame([{
                "date": pd.Timestamp("2026-10-22"),
                "group": "XLK", "group_type": "sector_usa",
                "ticker": "AAPL", "name": "AAPL", "weight": 50.0,
                "phase": "ACCUMULATION", "event_phase": "ACCUMULATION",
                "phase_since": "", "close": 100.0, "basis": "",
                "coverage_weight_pct": 100.0,
            }])
    return out


def _capture_calls(monkeypatch):
    calls = {"persist": [], "skip": []}
    def fake_persist(name, df, **kw):
        calls["persist"].append(name)
    def fake_skip(name, reason, **kw):
        calls["skip"].append((name, reason))
    monkeypatch.setattr(wh, "persist", fake_persist)
    monkeypatch.setattr(wh, "skip", fake_skip)
    return calls


def test_skips_when_df_stocks_empty(monkeypatch):
    calls = _capture_calls(monkeypatch)
    out = wh.compute_wyckoff_holdings(pd.DataFrame(), _make_holdings())
    assert out is None
    assert len(calls["persist"]) == 0
    assert len(calls["skip"]) == 6
    assert all(r == "INPUT_UNAVAILABLE" for _, r in calls["skip"])
    assert set(n for n, _ in calls["skip"]) == set(ALL_NAMES)


def test_skips_when_holdings_empty(monkeypatch):
    calls = _capture_calls(monkeypatch)
    df = _make_df_stocks(["AAPL"])
    out = wh.compute_wyckoff_holdings(df, pd.DataFrame())
    assert out is None
    assert len(calls["persist"]) == 0
    assert len(calls["skip"]) == 6
    assert all(r == "INPUT_UNAVAILABLE" for _, r in calls["skip"])


def test_persists_all_when_data_present(monkeypatch, tmp_path):
    calls = _capture_calls(monkeypatch)
    monkeypatch.setattr(
        wh, "compute_wyckoff_holdings_report",
        lambda *a, **k: _make_tables(),
    )
    out = wh.compute_wyckoff_holdings(
        _make_df_stocks(["AAPL"]),
        _make_holdings(),
        index_holdings_path=str(tmp_path / "nonexistent.csv"),
    )
    assert out is not None
    assert set(calls["persist"]) == set(ALL_NAMES)
    assert len(calls["skip"]) == 0


def test_skip_individual_empty_df(monkeypatch, tmp_path):
    calls = _capture_calls(monkeypatch)
    tables = _make_tables(empty_keys=(
        "usa_index_acc", "usa_index_dist", "eu_index_acc", "eu_index_dist",
    ))
    monkeypatch.setattr(wh, "compute_wyckoff_holdings_report", lambda *a, **k: tables)
    out = wh.compute_wyckoff_holdings(
        _make_df_stocks(["AAPL"]),
        _make_holdings(),
        index_holdings_path=str(tmp_path / "nonexistent.csv"),
    )
    assert out is not None
    assert set(calls["persist"]) == {
        "wyckoff_usa_sector_accumulation",
        "wyckoff_usa_sector_distribution",
    }
    empty_skips = [n for n, r in calls["skip"] if r == "EMPTY_DF"]
    assert len(empty_skips) == 4


def test_compute_error_emits_skip_for_all(monkeypatch, tmp_path):
    calls = _capture_calls(monkeypatch)
    def raiser(*a, **k):
        raise ValueError("boom")
    monkeypatch.setattr(wh, "compute_wyckoff_holdings_report", raiser)
    out = wh.compute_wyckoff_holdings(
        _make_df_stocks(["AAPL"]),
        _make_holdings(),
        index_holdings_path=str(tmp_path / "nonexistent.csv"),
    )
    assert out is None
    assert len(calls["persist"]) == 0
    assert len(calls["skip"]) == 6
    assert all(r == "COMPUTE_ERROR" for _, r in calls["skip"])


def test_load_index_holdings_missing_path(tmp_path):
    out = wh._load_index_holdings(str(tmp_path / "missing.csv"))
    assert out is None


def test_load_index_holdings_empty_csv(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("etf,ticker\n", encoding="utf-8")
    out = wh._load_index_holdings(str(p))
    assert out is None


def test_split_indices_filters_correctly():
    df = _make_index_holdings()
    usa, eu = wh._split_indices(df)
    assert usa is not None and eu is not None
    assert set(usa["etf"]) == {"SPY"}
    assert set(eu["etf"]) == {"DAXEX"}


def test_split_indices_returns_none_when_no_match():
    df = pd.DataFrame({"etf": ["IWM"], "ticker": ["X"]})
    usa, eu = wh._split_indices(df)
    assert usa is None
    assert eu is None