# -*- coding: utf-8 -*-
"""FIX-WYCKOFF-SIGN-ROUNDING (2026-11-14): _wyckoff_sign usa counts.

Bug: la regla documentada compara (Acc + Markup) vs (Dist + Markdown).
Sobre pct_* redondeados a 8 decimales, un empate real en counts puede
flipar el signo por artefacto de redondeo (diff 1e-8). XLY 2026-11-12
fue el caso: 2 vs 2 real → reportado -1.
"""
import numpy as np
import pandas as pd

from indicators.evidence_matrix import _wyckoff_sign


def test_empate_exacto_counts_devuelve_0():
    # XLY 2026-11-12 (valores reales del CSV): counts empate 2/2
    # pct_* redondeados flipan a -1 (4.25531914 < 4.25531915).
    row = {
        'count_accumulation': 1, 'count_markup': 1,
        'count_distribution': 0, 'count_markdown': 2,
        'pct_accumulation': 2.12765957, 'pct_markup': 2.12765957,
        'pct_distribution': 0.0, 'pct_markdown': 4.25531915,
    }
    assert _wyckoff_sign(row) == 0


def test_counts_favorables_devuelve_1():
    row = {'count_accumulation': 2, 'count_markup': 1,
           'count_distribution': 0, 'count_markdown': 0,
           'pct_accumulation': 2.0, 'pct_markup': 1.0,
           'pct_distribution': 0.0, 'pct_markdown': 0.0}
    assert _wyckoff_sign(row) == 1


def test_counts_desfavorables_devuelve_menos1():
    row = {'count_accumulation': 0, 'count_markup': 0,
           'count_distribution': 1, 'count_markdown': 2,
           'pct_accumulation': 0.0, 'pct_markup': 0.0,
           'pct_distribution': 1.0, 'pct_markdown': 2.0}
    assert _wyckoff_sign(row) == -1


def test_sin_counts_fallback_a_pct():
    # Compat con CSV pre-refactor sin counts.
    row = {'pct_accumulation': 5.0, 'pct_markup': 10.0,
           'pct_distribution': 3.0, 'pct_markdown': 2.0}
    assert _wyckoff_sign(row) == 1


def test_sin_counts_ni_pct_devuelve_nan():
    row = {}
    assert pd.isna(_wyckoff_sign(row))


def test_counts_nan_fallback_a_pct():
    # Si algun count es NaN, fallback a pct (compat).
    row = {'count_accumulation': np.nan, 'count_markup': 1,
           'count_distribution': 0, 'count_markdown': 2,
           'pct_accumulation': 5.0, 'pct_markup': 10.0,
           'pct_distribution': 3.0, 'pct_markdown': 2.0}
    assert _wyckoff_sign(row) == 1
