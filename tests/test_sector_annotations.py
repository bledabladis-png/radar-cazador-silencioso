"""Tests de anotacion de titulos sectoriales con composicion interna."""
from __future__ import annotations

import pandas as pd

from src.report.sector_annotations import (
    annotate_sector_titles_with_composition, _composition_line,
)


def _make_dist_df(rows):
    """rows: dict sector -> dict de counts/pcts"""
    return pd.DataFrame(rows)


def test_annotate_simple():
    leader_lines = [
        "## Sector: XLK (MARKUP)\n",
        "| Ticker | RS |\n",
        "|--------|----|\n",
        "| AAPL | 1.0 |\n",
        "\n",
    ]
    dist_df = pd.DataFrame([{
        "sector": "XLK", "date": "2026-10-23",
        "n_valid_wyckoff": 22,
        "count_accumulation": 0, "pct_accumulation": 0.0,
        "count_accumulation_forming": 0, "pct_accumulation_forming": 0.0,
        "count_markup": 3, "pct_markup": 14.0,
        "count_range": 8, "pct_range": 36.0,
        "count_distribution_forming": 9, "pct_distribution_forming": 41.0,
        "count_distribution": 1, "pct_distribution": 5.0,
        "count_markdown": 1, "pct_markdown": 5.0,
        "count_conflict": 0, "pct_conflict": 0.0,
    }])
    result = annotate_sector_titles_with_composition(leader_lines, dist_df)
    content = "".join(result)
    assert "## Sector: XLK (MARKUP)" in content
    assert "Composicion interna (22 componentes)" in content
    assert "9 DistF (41%)" in content
    assert "3 Markup (14%)" in content


def test_annotate_none_returns_unchanged():
    lines = ["## Sector: XLK (MARKUP)\n"]
    assert annotate_sector_titles_with_composition(lines, None) is lines


def test_annotate_empty_df_returns_unchanged():
    lines = ["## Sector: XLK (MARKUP)\n"]
    assert annotate_sector_titles_with_composition(lines, pd.DataFrame()) is lines


def test_annotate_sector_not_in_df():
    leader_lines = ["## Sector: ZZZ (MARKUP)\n", "rest\n"]
    dist_df = pd.DataFrame([{"sector": "XLK", "n_valid_wyckoff": 20,
                             "count_markup": 20, "pct_markup": 100.0}])
    result = annotate_sector_titles_with_composition(leader_lines, dist_df)
    content = "".join(result)
    assert content == "## Sector: ZZZ (MARKUP)\nrest\n"


def test_annotate_idempotent():
    leader_lines = [
        "## Sector: XLK (MARKUP)\n",
        "\n",
        "*Composicion interna (22 componentes): 3 Markup (14%)*\n",
        "\n",
        "rest\n",
    ]
    dist_df = pd.DataFrame([{"sector": "XLK", "n_valid_wyckoff": 22,
                             "count_markup": 3, "pct_markup": 14.0}])
    result = annotate_sector_titles_with_composition(leader_lines, dist_df)
    content = "".join(result)
    # No debe duplicar
    assert content.count("Composicion interna") == 1


def test_annotate_zero_components_skips():
    leader_lines = ["## Sector: XLK (MARKUP)\n", "rest\n"]
    dist_df = pd.DataFrame([{"sector": "XLK", "n_valid_wyckoff": 0,
                             "count_markup": 0, "pct_markup": 0.0}])
    result = annotate_sector_titles_with_composition(leader_lines, dist_df)
    assert "Composicion interna" not in "".join(result)


def test_composition_line_skips_zero_phases():
    row = pd.Series({
        "n_valid_wyckoff": 10,
        "count_accumulation": 0, "pct_accumulation": 0.0,
        "count_markup": 5, "pct_markup": 50.0,
        "count_range": 5, "pct_range": 50.0,
        "count_distribution": 0, "pct_distribution": 0.0,
        "count_distribution_forming": 0, "pct_distribution_forming": 0.0,
        "count_markdown": 0, "pct_markdown": 0.0,
        "count_accumulation_forming": 0, "pct_accumulation_forming": 0.0,
        "count_conflict": 0, "pct_conflict": 0.0,
    })
    line = _composition_line(row)
    assert line is not None
    assert "5 Markup" in line
    assert "5 Range" in line
    assert "0 " not in line.replace("10 ", "")  # no debe incluir ceros