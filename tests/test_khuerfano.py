# -*- coding: utf-8 -*-
"""Test K-HUERFANO: deteccion de lote parcialmente incompleto (dictamen 2026-09-17)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import _check_khuerfano


def _make_batch(tickers, dates, values):
    idx = pd.DatetimeIndex(dates)
    cols = pd.MultiIndex.from_tuples([("Close", t) for t in tickers])
    return pd.DataFrame(values, index=idx, columns=cols)


class TestCheckKhuerfano:
    def test_batch_completo_sin_warning(self):
        df = _make_batch(["AAA", "BBB"], ["2026-09-15", "2026-09-16"],
                         [[10.0, 20.0], [11.0, 21.0]])
        assert _check_khuerfano(df, ["AAA", "BBB"], "2026-09-16") == []

    def test_batch_parcial_detecta_khuerfano(self):
        df = _make_batch(["AAA", "KHC"], ["2026-09-15", "2026-09-16"],
                         [[10.0, 20.0], [11.0, np.nan]])
        assert _check_khuerfano(df, ["AAA", "KHC"], "2026-09-16") == ["KHC"]

    def test_nan_fuera_de_expected_no_alerta(self):
        df = _make_batch(["AAA"], ["2026-09-14", "2026-09-15"],
                         [[10.0], [np.nan]])
        assert _check_khuerfano(df, ["AAA"], "2026-09-16") == []

    def test_expected_session_none_no_alerta(self):
        df = _make_batch(["AAA"], ["2026-09-15", "2026-09-16"],
                         [[10.0], [np.nan]])
        assert _check_khuerfano(df, ["AAA"], None) == []

    def test_dataframe_vacio_no_alerta(self):
        assert _check_khuerfano(pd.DataFrame(), ["AAA"], "2026-09-16") == []

    def test_ticker_sin_columna_no_alerta(self):
        cols = pd.MultiIndex.from_tuples([("Close", "AAA")])
        df = pd.DataFrame([[10.0]], index=pd.DatetimeIndex(["2026-09-16"]), columns=cols)
        assert _check_khuerfano(df, ["AAA", "BBB"], "2026-09-16") == []