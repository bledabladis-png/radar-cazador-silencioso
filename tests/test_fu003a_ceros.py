# tests/test_fu003a_ceros.py
"""Tests FU-003a (2026-09-15): cero sin signo con _fmt_signed."""
import numpy as np
import pandas as pd

from src.report.helpers import _fmt_signed


def test_fmt_signed_positive():
    assert _fmt_signed(0.42, '{:+.2f}', '{:.2f}') == '+0.42'


def test_fmt_signed_negative():
    assert _fmt_signed(-0.42, '{:+.2f}', '{:.2f}') == '-0.42'


def test_fmt_signed_zero_no_sign():
    """FU-003a: cero NO debe llevar '+'."""
    assert _fmt_signed(0.0, '{:+.2f}', '{:.2f}') == '0.00'
    assert _fmt_signed(-0.0, '{:+.2f}', '{:.2f}') == '0.00'
    assert _fmt_signed(0, '{:+.2f}', '{:.2f}') == '0.00'


def test_fmt_signed_nan():
    assert _fmt_signed(np.nan, '{:+.2f}', '{:.2f}') == 'N/D'
    assert _fmt_signed(pd.NA, '{:+.2f}', '{:.2f}') == 'N/D'
    assert _fmt_signed(None, '{:+.2f}', '{:.2f}') == 'N/D'


def test_fmt_signed_pct_zero():
    """Formato con % y cero -> sin signo."""
    assert _fmt_signed(0.0, '{:+.6f}%', '{:.6f}%') == '0.000000%'
    assert _fmt_signed(0.05, '{:+.6f}%', '{:.6f}%') == '+0.050000%'


def test_fmt_signed_thousands_sep():
    assert _fmt_signed(0.0, '{:+,.2f}', '{:,.2f}') == '0.00'
    assert _fmt_signed(1234.56, '{:+,.2f}', '{:,.2f}') == '+1,234.56'


def test_fmt_signed_very_small_displays_as_zero():
    """FU-003a (refinado): valores que muestran '0.00' van sin signo.

    Incluye ruido flotante (1e-9) y valores absolutos menores que la
    precision de display (0.004 con 2 decimales).
    """
    assert _fmt_signed(1e-9, '{:+.2f}', '{:.2f}') == '0.00'
    assert _fmt_signed(0.004, '{:+.2f}', '{:.2f}') == '0.00'
    assert _fmt_signed(-0.004, '{:+.2f}', '{:.2f}') == '0.00'
    # Valores que SI muestran digitos no-cero mantienen signo.
    assert _fmt_signed(0.006, '{:+.2f}', '{:.2f}') == '+0.01'
    assert _fmt_signed(-0.006, '{:+.2f}', '{:.2f}') == '-0.01'
