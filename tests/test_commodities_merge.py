"""Tests de src/commodities_merge.py (FU-021-3C-bis).

Sin red. Sin dependencia de data/ real. Todos los parquets son
temporales (tmp_path).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.commodities_merge import merge_commodities_into_market


def _mk_market(dates, tickers_values):
    """Construye df_market con MultiIndex (field, ticker)."""
    idx = pd.to_datetime(dates)
    cols = pd.MultiIndex.from_tuples(
        [("Close", t) for t in tickers_values.keys()],
        names=["field", "ticker"],
    )
    df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for t, vals in tickers_values.items():
        df[("Close", t)] = vals
    return df


def _mk_comm(dates, tickers_values):
    idx = pd.to_datetime(dates)
    cols = pd.MultiIndex.from_tuples(
        [("Close", t) for t in tickers_values.keys()],
        names=["field", "ticker"],
    )
    df = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for t, vals in tickers_values.items():
        df[("Close", t)] = vals
    return df


class TestMergeBasic:

    def test_sobrescribe_en_fechas_comunes(self, tmp_path):
        df = _mk_market(
            ["2026-09-13", "2026-09-14", "2026-09-15"],
            {"BZ=F": [10.0, 20.0, 30.0]},
        )
        comm = _mk_comm(["2026-09-14"], {"BZ=F": [108.32]})
        p = tmp_path / "fut.parquet"
        comm.to_parquet(p)

        out = merge_commodities_into_market(df, futures_path=str(p), spot_path=None)

        assert out.loc["2026-09-13", ("Close", "BZ=F")] == 10.0
        assert out.loc["2026-09-14", ("Close", "BZ=F")] == pytest.approx(108.32)
        assert out.loc["2026-09-15", ("Close", "BZ=F")] == 30.0

    def test_anade_columna_nueva(self, tmp_path):
        df = _mk_market(
            ["2026-09-14"], {"BZ=F": [10.0]},
        )
        comm = _mk_comm(["2026-09-14"], {"GC=F": [4341.77]})
        p = tmp_path / "spot.parquet"
        comm.to_parquet(p)

        out = merge_commodities_into_market(df, futures_path=None, spot_path=str(p))

        assert ("Close", "GC=F") in out.columns
        assert out.loc["2026-09-14", ("Close", "GC=F")] == pytest.approx(4341.77)
        assert out.loc["2026-09-14", ("Close", "BZ=F")] == 10.0

    def test_sin_fechas_comunes_no_cambia(self, tmp_path):
        df = _mk_market(["2026-09-14"], {"BZ=F": [10.0]})
        comm = _mk_comm(["2026-09-10"], {"BZ=F": [99.0]})
        p = tmp_path / "fut.parquet"
        comm.to_parquet(p)

        out = merge_commodities_into_market(df, futures_path=str(p), spot_path=None)

        assert out.loc["2026-09-14", ("Close", "BZ=F")] == 10.0
        assert len(out) == 1

    def test_no_anade_filas_a_market(self, tmp_path):
        df = _mk_market(["2026-09-14"], {"BZ=F": [10.0]})
        comm = _mk_comm(
            ["2026-09-13", "2026-09-14", "2026-09-15"],
            {"BZ=F": [1.0, 2.0, 3.0]},
        )
        p = tmp_path / "fut.parquet"
        comm.to_parquet(p)

        out = merge_commodities_into_market(df, futures_path=str(p), spot_path=None)

        assert len(out) == 1
        assert out.index[0] == pd.Timestamp("2026-09-14")


class TestMergeDegradacion:

    def test_parquets_no_existen_no_falla(self):
        df = _mk_market(["2026-09-14"], {"BZ=F": [10.0]})
        out = merge_commodities_into_market(
            df,
            futures_path="/ruta/inexistente/fut.parquet",
            spot_path="/ruta/inexistente/spot.parquet",
        )
        assert out is df
        assert out.loc["2026-09-14", ("Close", "BZ=F")] == 10.0

    def test_df_market_vacio(self, tmp_path):
        df = pd.DataFrame()
        out = merge_commodities_into_market(df, futures_path=None, spot_path=None)
        assert out is df

    def test_df_market_sin_multiindex(self, tmp_path):
        df = pd.DataFrame({"BZ=F": [10.0]}, index=pd.to_datetime(["2026-09-14"]))
        comm = _mk_comm(["2026-09-14"], {"BZ=F": [99.0]})
        p = tmp_path / "fut.parquet"
        comm.to_parquet(p)

        out = merge_commodities_into_market(df, futures_path=str(p), spot_path=None)

        assert out.loc["2026-09-14", "BZ=F"] == 10.0

    def test_parquet_sin_multiindex_se_ignora(self, tmp_path):
        df = _mk_market(["2026-09-14"], {"BZ=F": [10.0]})
        bad = pd.DataFrame({"BZ=F": [99.0]}, index=pd.to_datetime(["2026-09-14"]))
        p = tmp_path / "bad.parquet"
        bad.to_parquet(p)

        out = merge_commodities_into_market(df, futures_path=str(p), spot_path=None)

        assert out.loc["2026-09-14", ("Close", "BZ=F")] == 10.0


class TestMergeFuturosYSpot:

    def test_ambos_parquets_se_mergean(self, tmp_path):
        df = _mk_market(
            ["2026-09-14"],
            {"BZ=F": [10.0], "GC=F": [0.0]},
        )
        fut = _mk_comm(["2026-09-14"], {"BZ=F": [108.32]})
        spot = _mk_comm(["2026-09-14"], {"GC=F": [4341.77]})
        p_fut = tmp_path / "fut.parquet"
        p_spot = tmp_path / "spot.parquet"
        fut.to_parquet(p_fut)
        spot.to_parquet(p_spot)

        out = merge_commodities_into_market(
            df, futures_path=str(p_fut), spot_path=str(p_spot)
        )

        assert out.loc["2026-09-14", ("Close", "BZ=F")] == pytest.approx(108.32)
        assert out.loc["2026-09-14", ("Close", "GC=F")] == pytest.approx(4341.77)

    def test_idempotencia(self, tmp_path):
        df = _mk_market(["2026-09-14"], {"BZ=F": [10.0]})
        comm = _mk_comm(["2026-09-14"], {"BZ=F": [108.32]})
        p = tmp_path / "fut.parquet"
        comm.to_parquet(p)

        out1 = merge_commodities_into_market(df, futures_path=str(p), spot_path=None)
        v1 = out1.loc["2026-09-14", ("Close", "BZ=F")]

        out2 = merge_commodities_into_market(df, futures_path=str(p), spot_path=None)
        v2 = out2.loc["2026-09-14", ("Close", "BZ=F")]

        assert v1 == v2 == pytest.approx(108.32)
