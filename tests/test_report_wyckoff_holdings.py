"""Tests de src.report.wyckoff_holdings."""
import pandas as pd

from src.report import wyckoff_holdings as rh


COLS = [
    "date", "group", "group_type", "ticker", "name", "weight",
    "phase", "event_phase", "phase_since", "close", "basis",
    "coverage_weight_pct",
]


def _make_df(rows):
    return pd.DataFrame(rows, columns=COLS)


def _row(group, ticker, phase, weight=50.0, close=100.0,
         since="2026-10-15", cov=80.0, group_type="sector_usa"):
    return {
        "date": pd.Timestamp("2026-10-22"),
        "group": group, "group_type": group_type,
        "ticker": ticker, "name": ticker,
        "weight": weight, "phase": phase,
        "event_phase": phase, "phase_since": since,
        "close": close, "basis": "",
        "coverage_weight_pct": cov,
    }


def test_empty_dict_returns_empty():
    assert rh.render_wyckoff_holdings_acc(None) == []
    assert rh.render_wyckoff_holdings_acc({}) == []
    assert rh.render_wyckoff_holdings_dist(None) == []


def test_all_empty_tables_returns_empty():
    empty = pd.DataFrame(columns=COLS)
    tables = {k: empty for k in rh.ACC_KEYS}
    assert rh.render_wyckoff_holdings_acc(tables) == []


def test_one_subsection_only():
    tables = {
        "usa_sector_acc": _make_df([
            _row("XLK", "AAPL", "ACCUMULATION"),
        ]),
    }
    out = rh.render_wyckoff_holdings_acc(tables)
    text = "".join(out)
    assert "## Tickers en acumulacion" in text
    assert "### USA sectores" in text
    assert "AAPL" in text
    assert "### USA indices" in text
    assert "### Indices europeos" in text


def test_dist_section_uses_dist_keys():
    tables = {
        "usa_sector_dist": _make_df([
            _row("XLK", "XYZ", "DISTRIBUTION_FORMING"),
        ]),
    }
    out = rh.render_wyckoff_holdings_dist(tables)
    text = "".join(out)
    assert "## Tickers en distribucion" in text
    assert "XYZ" in text
    assert "DISTRIBUTION_FORMING" in text


def test_empty_subsection_has_nota():
    tables = {
        "usa_sector_acc": _make_df([
            _row("XLK", "AAPL", "MARKUP"),
        ]),
    }
    out = rh.render_wyckoff_holdings_acc(tables)
    text = "".join(out)
    assert "*Sin tickers en este estado.*" in text


def test_summary_counts_by_phase():
    tables = {
        "usa_sector_acc": _make_df([
            _row("XLK", "T1", "ACCUMULATION"),
            _row("XLK", "T2", "ACCUMULATION"),
            _row("XLK", "T3", "MARKUP"),
        ]),
    }
    out = rh.render_wyckoff_holdings_acc(tables)
    text = "".join(out)
    assert "3 tickers" in text
    assert "2 ACCUMULATION" in text
    assert "1 MARKUP" in text


def test_missing_phase_since_formats_as_dash():
    tables = {
        "usa_sector_acc": _make_df([
            _row("XLK", "T1", "ACCUMULATION", since=""),
        ]),
    }
    out = rh.render_wyckoff_holdings_acc(tables)
    text = "".join(out)
    assert "| - |" in text


def test_nan_values_format_as_dash():
    r = _row("XLK", "T1", "ACCUMULATION")
    r["close"] = float("nan")
    r["weight"] = float("nan")
    tables = {"usa_sector_acc": _make_df([r])}
    out = rh.render_wyckoff_holdings_acc(tables)
    text = "".join(out)
    assert "nan" not in text.lower()


def test_both_sections_independent():
    tables = {
        "usa_sector_acc": _make_df([_row("XLK", "AAPL", "ACCUMULATION")]),
        "usa_sector_dist": _make_df([_row("XLK", "XYZ", "DISTRIBUTION")]),
    }
    acc_text = "".join(rh.render_wyckoff_holdings_acc(tables))
    dist_text = "".join(rh.render_wyckoff_holdings_dist(tables))
    assert "AAPL" in acc_text and "XYZ" not in acc_text
    assert "XYZ" in dist_text and "AAPL" not in dist_text