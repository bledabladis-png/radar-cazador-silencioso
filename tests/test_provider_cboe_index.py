"""Tests unitarios del provider CBOE Index (FU-021-3D).

Sin red. Todo mockeado. Cubre: parsing CSV -> DataFrame wide,
merge con parquet previo (dedupe), politica de reintentos.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.providers.cboe_index import (
    CboeIndexProvider,
    _merge_with_existing,
    _parse_csv_to_wide,
)


# --- Fixtures CSV simulados --------------------------------------

CSV_OK = """DATE,OPEN,HIGH,LOW,CLOSE
2026-09-11,20.10,20.55,19.80,20.30
2026-09-14,20.25,20.70,20.00,20.55
2026-09-15,20.40,20.85,20.15,20.70
"""

CSV_FECHA_INVALIDA = """DATE,OPEN,HIGH,LOW,CLOSE
2026-09-14,20.25,20.70,20.00,20.55
no-fecha,1.0,1.0,1.0,1.0
2026-09-15,20.40,20.85,20.15,20.70
"""

CSV_SIN_CLOSE = """DATE,OPEN,HIGH,LOW
2026-09-15,20.40,20.85,20.15
"""

CSV_SIN_DATE = """OPEN,HIGH,LOW,CLOSE
20.40,20.85,20.15,20.70
"""


# --- Parsing CSV -> wide -----------------------------------------

class TestParseCsvToWide:

    def test_estructura_multiindex(self):
        df = _parse_csv_to_wide(CSV_OK, ticker="^VIX3M")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert isinstance(df.columns, pd.MultiIndex)
        assert ("Close", "^VIX3M") in df.columns
        assert ("Open", "^VIX3M") in df.columns
        assert ("High", "^VIX3M") in df.columns
        assert ("Low", "^VIX3M") in df.columns
        assert ("Volume", "^VIX3M") in df.columns

    def test_valores(self):
        df = _parse_csv_to_wide(CSV_OK, ticker="^VIX3M")
        assert df.loc["2026-09-15", ("Close", "^VIX3M")] == pytest.approx(20.70)
        assert df.loc["2026-09-11", ("Open", "^VIX3M")] == pytest.approx(20.10)

    def test_ordenado(self):
        df = _parse_csv_to_wide(CSV_OK, ticker="^VIX3M")
        assert df.index.is_monotonic_increasing

    def test_volume_nan(self):
        df = _parse_csv_to_wide(CSV_OK, ticker="^VIX3M")
        assert df[("Volume", "^VIX3M")].isna().all()

    def test_fecha_invalida_descartada(self):
        df = _parse_csv_to_wide(CSV_FECHA_INVALIDA, ticker="^VIX3M")
        assert len(df) == 2
        assert pd.Timestamp("2026-09-15") in df.index

    def test_sin_close_genera_columna_nan(self):
        df = _parse_csv_to_wide(CSV_SIN_CLOSE, ticker="^VIX3M")
        assert ("Close", "^VIX3M") in df.columns
        assert df[("Close", "^VIX3M")].isna().all()

    def test_sin_date_lanza(self):
        with pytest.raises(ValueError, match="DATE"):
            _parse_csv_to_wide(CSV_SIN_DATE, ticker="^VIX3M")


# --- Fetch (mockeado) --------------------------------------------

class TestFetchVix3m:

    def test_ok(self, monkeypatch):
        p = CboeIndexProvider()
        monkeypatch.setattr(p, "_get_csv", lambda: CSV_OK)
        df = p.fetch_vix3m()
        assert not df.empty
        assert ("Close", "^VIX3M") in df.columns
        assert df.loc["2026-09-15", ("Close", "^VIX3M")] == pytest.approx(20.70)

    def test_error_devuelve_vacio(self, monkeypatch):
        p = CboeIndexProvider()
        def boom():
            raise RuntimeError("network fail")
        monkeypatch.setattr(p, "_get_csv", boom)
        df = p.fetch_vix3m()
        assert df.empty


# --- Merge con existente -----------------------------------------

class TestMergeWithExisting:

    def test_sin_fichero_previo(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        out = _merge_with_existing(df, str(tmp_path / "no_existe.parquet"))
        assert out is df

    def test_dedupe_keep_last(self, tmp_path):
        p = tmp_path / "x.parquet"
        existing = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.to_datetime(["2026-09-14", "2026-09-15"]),
        )
        existing.to_parquet(p)

        new = pd.DataFrame(
            {"Close": [999.0]},
            index=pd.to_datetime(["2026-09-15"]),
        )
        out = _merge_with_existing(new, str(p))
        assert len(out) == 2
        assert out.loc["2026-09-15", "Close"] == 999.0
        assert out.loc["2026-09-14", "Close"] == 100.0

    def test_fichero_vacio(self, tmp_path):
        p = tmp_path / "empty.parquet"
        pd.DataFrame().to_parquet(p)
        df = pd.DataFrame({"a": [1]})
        out = _merge_with_existing(df, str(p))
        assert out is df


# --- Politica de reintentos --------------------------------------

class TestRetryPolicy:

    def test_4xx_no_reintenta(self, monkeypatch):
        class R:
            status_code = 404
            text = "not found"

        calls = {"n": 0}
        def fake_get(*a, **kw):
            calls["n"] += 1
            return R()

        monkeypatch.setattr(
            "data.providers.cboe_index.requests.Session.get", fake_get
        )
        p = CboeIndexProvider()
        with pytest.raises(RuntimeError, match="404"):
            p._get_csv()
        assert calls["n"] == 1


# --- Interfaz provider -------------------------------------------

class TestProviderInterface:

    def test_name(self):
        assert CboeIndexProvider().get_name() == "CBOE Index"

    def test_get_prices_no_implementado(self):
        with pytest.raises(NotImplementedError):
            CboeIndexProvider().get_prices(["^VIX3M"])

    def test_get_treasury_yields_no_implementado(self):
        with pytest.raises(NotImplementedError):
            CboeIndexProvider().get_treasury_yields()

    def test_get_fed_data_no_implementado(self):
        with pytest.raises(NotImplementedError):
            CboeIndexProvider().get_fed_data()