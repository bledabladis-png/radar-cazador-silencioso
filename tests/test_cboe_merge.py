"""Tests de src/cboe_merge.py (FU-021-3D).

Sin red. Sin dependencia de data/ real. Todos los parquets son
temporales (tmp_path).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.cboe_merge import merge_cboe_into_market


def _mk_market(dates, ticker_values):
    """Construye df_market con MultiIndex (field, ticker)."""
    idx = pd.to_datetime(dates)
    cols = pd.MultiIndex.from_tuples(
        [("Close", t) for t in ticker_values.keys()],
        names=["field", "ticker"],
    )
    df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for t, vals in ticker_values.items():
        df[("Close", t)] = vals
    return df


def _mk_cboe(dates, values):
    """Construye parquet CBOE con MultiIndex (field, ticker)."""
    idx = pd.to_datetime(dates)
    fields = ["Open", "High", "Low", "Close", "Volume"]
    cols = pd.MultiIndex.from_tuples(
        [(f, "^VIX3M") for f in fields],
        names=["field", "ticker"],
    )
    df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for f in ("Open", "High", "Low", "Close"):
        df[(f, "^VIX3M")] = values
    df[("Volume", "^VIX3M")] = float("nan")
    return df


class TestMergeBasic:

    def test_sobrescribe_en_fechas_comunes(self, tmp_path):
        df = _mk_market(
            ["2026-09-13", "2026-09-14", "2026-09-15"],
            {"^VIX3M": [float("nan")] * 3},
        )
        cboe = _mk_cboe(["2026-09-14"], [20.55])
        p = tmp_path / "cboe.parquet"
        cboe.to_parquet(p)

        out = merge_cboe_into_market(df, cboe_path=str(p))

        assert pd.isna(out.loc["2026-09-13", ("Close", "^VIX3M")])
        assert out.loc["2026-09-14", ("Close", "^VIX3M")] == pytest.approx(20.55)
        assert pd.isna(out.loc["2026-09-15", ("Close", "^VIX3M")])

    def test_anade_columnas_open_high_low_volume(self, tmp_path):
        df = _mk_market(["2026-09-14"], {"^VIX3M": [float("nan")]})
        cboe = _mk_cboe(["2026-09-14"], [20.55])
        p = tmp_path / "cboe.parquet"
        cboe.to_parquet(p)

        out = merge_cboe_into_market(df, cboe_path=str(p))

        assert ("Open", "^VIX3M") in out.columns
        assert ("High", "^VIX3M") in out.columns
        assert ("Low", "^VIX3M") in out.columns
        assert ("Volume", "^VIX3M") in out.columns
        assert out.loc["2026-09-14", ("Open", "^VIX3M")] == pytest.approx(20.55)

    def test_anade_ticker_nuevo_si_no_existe(self, tmp_path):
        df = _mk_market(["2026-09-14"], {"^VIX": [15.0]})
        cboe = _mk_cboe(["2026-09-14"], [20.55])
        p = tmp_path / "cboe.parquet"
        cboe.to_parquet(p)

        out = merge_cboe_into_market(df, cboe_path=str(p))

        assert ("Close", "^VIX3M") in out.columns
        assert out.loc["2026-09-14", ("Close", "^VIX3M")] == pytest.approx(20.55)
        assert out.loc["2026-09-14", ("Close", "^VIX")] == 15.0

    def test_sin_fechas_comunes_no_cambia(self, tmp_path):
        df = _mk_market(["2026-09-14"], {"^VIX3M": [float("nan")]})
        cboe = _mk_cboe(["2026-09-10"], [99.0])
        p = tmp_path / "cboe.parquet"
        cboe.to_parquet(p)

        out = merge_cboe_into_market(df, cboe_path=str(p))

        assert pd.isna(out.loc["2026-09-14", ("Close", "^VIX3M")])
        assert len(out) == 1

    def test_no_anade_filas_a_market(self, tmp_path):
        df = _mk_market(["2026-09-14"], {"^VIX3M": [float("nan")]})
        cboe = _mk_cboe(
            ["2026-09-13", "2026-09-14", "2026-09-15"],
            [1.0, 2.0, 3.0],
        )
        p = tmp_path / "cboe.parquet"
        cboe.to_parquet(p)

        out = merge_cboe_into_market(df, cboe_path=str(p))

        assert len(out) == 1
        assert out.index[0] == pd.Timestamp("2026-09-14")


class TestMergeDegradacion:

    def test_parquet_no_existe_no_falla(self):
        df = _mk_market(["2026-09-14"], {"^VIX3M": [float("nan")]})
        out = merge_cboe_into_market(
            df, cboe_path="/ruta/inexistente/cboe.parquet"
        )
        assert out is df

    def test_df_market_vacio(self):
        df = pd.DataFrame()
        out = merge_cboe_into_market(df, cboe_path="x.parquet")
        assert out is df

    def test_df_market_sin_multiindex(self, tmp_path):
        df = pd.DataFrame(
            {"^VIX3M": [float("nan")]},
            index=pd.to_datetime(["2026-09-14"]),
        )
        cboe = _mk_cboe(["2026-09-14"], [20.55])
        p = tmp_path / "cboe.parquet"
        cboe.to_parquet(p)

        out = merge_cboe_into_market(df, cboe_path=str(p))

        assert pd.isna(out.loc["2026-09-14", "^VIX3M"])

    def test_parquet_sin_multiindex_se_ignora(self, tmp_path):
        df = _mk_market(["2026-09-14"], {"^VIX3M": [float("nan")]})
        bad = pd.DataFrame(
            {"^VIX3M": [99.0]},
            index=pd.to_datetime(["2026-09-14"]),
        )
        p = tmp_path / "bad.parquet"
        bad.to_parquet(p)

        out = merge_cboe_into_market(df, cboe_path=str(p))

        assert pd.isna(out.loc["2026-09-14", ("Close", "^VIX3M")])


class TestMergeIdempotencia:

    def test_idempotencia(self, tmp_path):
        df = _mk_market(["2026-09-14"], {"^VIX3M": [float("nan")]})
        cboe = _mk_cboe(["2026-09-14"], [20.55])
        p = tmp_path / "cboe.parquet"
        cboe.to_parquet(p)

        out1 = merge_cboe_into_market(df, cboe_path=str(p))
        v1 = out1.loc["2026-09-14", ("Close", "^VIX3M")]

        out2 = merge_cboe_into_market(df, cboe_path=str(p))
        v2 = out2.loc["2026-09-14", ("Close", "^VIX3M")]

        assert v1 == v2 == pytest.approx(20.55)

    def test_segunda_llamada_mantiene_shape(self, tmp_path):
        df = _mk_market(["2026-09-14"], {"^VIX3M": [float("nan")]})
        cboe = _mk_cboe(["2026-09-14"], [20.55])
        p = tmp_path / "cboe.parquet"
        cboe.to_parquet(p)

        shape0 = df.shape
        merge_cboe_into_market(df, cboe_path=str(p))
        shape1 = df.shape
        merge_cboe_into_market(df, cboe_path=str(p))
        shape2 = df.shape

        assert shape1[0] == shape0[0] == shape2[0]