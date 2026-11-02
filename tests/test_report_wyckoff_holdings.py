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


def test_min_weight_filter_excluye_bajo_umbral():
    """Filtro 0.5%: tickers con weight residual no aparecen en el .md."""
    tables = {
        "usa_sector_acc": _make_df([
            _row("SPY", "BIG", "MARKUP", weight=5.0),
            _row("SPY", "TINY", "MARKUP", weight=0.1),
        ]),
    }
    text = "".join(rh.render_wyckoff_holdings_acc(tables))
    assert "BIG" in text
    assert "TINY" not in text
    assert "Filtro: 1 de 2" in text


def test_min_weight_filter_no_aplica_si_todos_pasan():
    """Sin filtro efectivo: no aparece la nota de filtro."""
    tables = {
        "usa_sector_acc": _make_df([
            _row("XLK", "A", "MARKUP", weight=5.0),
            _row("XLK", "B", "MARKUP", weight=3.0),
        ]),
    }
    text = "".join(rh.render_wyckoff_holdings_acc(tables))
    assert "A" in text and "B" in text
    assert "Filtro:" not in text
    assert "2 tickers" in text


def test_min_weight_filter_todos_excluidos_fallback_visible():
    """Si todos caen bajo el umbral: nota visible, no silencio."""
    tables = {
        "usa_sector_acc": _make_df([
            _row("SPY", "TINY1", "MARKUP", weight=0.1),
            _row("SPY", "TINY2", "MARKUP", weight=0.2),
        ]),
    }
    text = "".join(rh.render_wyckoff_holdings_acc(tables))
    assert "0 de 2 tickers" in text
    assert "peso >= 0.50%" in text
    assert "TINY1" not in text


def test_min_weight_filter_nan_se_excluye():
    """weight NaN se trata como 0: excluido del .md."""
    r1 = _row("SPY", "GOOD", "MARKUP", weight=5.0)
    r2 = _row("SPY", "BAD", "MARKUP", weight=float("nan"))
    tables = {"usa_sector_acc": _make_df([r1, r2])}
    text = "".join(rh.render_wyckoff_holdings_acc(tables))
    assert "GOOD" in text
    assert "BAD" not in text


def test_fmt_num_nan_devuelve_dash():
    """Cobertura directa de _fmt_num(NaN): el filtro puede excluir
    filas antes, pero el helper debe seguir formateando NaN como dash."""
    assert rh._fmt_num(float("nan"), 2) == "-"
    assert rh._fmt_num(None, 2) == "-"
