# -*- coding: utf-8 -*-
"""Tests de src.formatting.format_signed_float."""
from __future__ import annotations

import pytest

from src.formatting import format_signed_float


class TestFormatSignedFloat:

    @pytest.mark.parametrize("value, expected", [
        (-0.001, "+0.00"),
        (-0.004, "+0.00"),
        (-0.0, "+0.00"),
        (0.0, "+0.00"),
        (0.001, "+0.00"),
        (0.004, "+0.00"),
        (-0.5, "-0.50"),
        (0.5, "+0.50"),
        (1.234, "+1.23"),
        (-1.234, "-1.23"),
        (10.0, "+10.00"),
        (-10.0, "-10.00"),
        (-123.456, "-123.46"),
    ])
    def test_casos(self, value, expected):
        assert format_signed_float(value) == expected

    def test_none(self):
        assert format_signed_float(None) == "N/A"

    def test_nan(self):
        assert format_signed_float(float("nan")) == "N/A"

    def test_inf(self):
        assert format_signed_float(float("inf")) == "N/A"
        assert format_signed_float(float("-inf")) == "N/A"

    def test_string_invalido(self):
        assert format_signed_float("abc") == "N/A"

    def test_decimals_override(self):
        assert format_signed_float(-0.0001, decimals=3) == "+0.000"
        assert format_signed_float(1.2345, decimals=4) == "+1.2345"

    def test_decimals_cero(self):
        assert format_signed_float(-0.4, decimals=0) == "+0"
        assert format_signed_float(-0.6, decimals=0) == "-1"
        assert format_signed_float(-0.001, decimals=0) == "+0"
