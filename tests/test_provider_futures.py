"""Tests unitarios del provider OilPriceAPI (FU-021-3C-bis).

Sin red. Todo mockeado. Cubre: parsing JSON -> DataFrame, merge con
parquet previo (dedupe), resolucion de API key.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.providers.futures import (
    FuturesProvider,
    _merge_with_existing,
    _rows_to_wide,
    _to_float,
)
from src.commodities_merge import merge_commodities_into_market


# --- Fixtures JSON simulados -------------------------------------

BRENT_JSON = {
    "commodity": "BRENT_FUTURES",
    "settlement_date": "2026-09-15",
    "front_month": {
        "code": "BRENT_FUTURES_2026_11",
        "contract_month": "2026-11",
        "open": "106.43",
        "high": "109.43",
        "low": "105.42",
        "close": "108.32",
        "volume": 561034,
    },
}

WTI_JSON = {
    "commodity": "WTI_FUTURES",
    "settlement_date": "2026-09-15",
    "front_month": {
        "code": "WTI_FUTURES_2026_11",
        "open": "70.10",
        "high": "72.50",
        "low": "69.80",
        "close": "71.90",
        "volume": 321000,
    },
}

SPOT_JSON = {
    "status": "ok",
    "data": {
        "prices": [
            {
                "code": "GOLD_USD",
                "price": 4341.77,
                "updated_at": "2026-09-16T13:52:53.051Z",
            },
            {
                "code": "COPPER_USD",
                "price": 6.37,
                "updated_at": "2026-09-16T13:52:53.051Z",
            },
            {
                "code": "NATURAL_GAS_USD",
                "price": 2.92,
                "updated_at": "2026-09-16T13:52:53.051Z",
            },
        ]
    },
}


# --- Parsing -----------------------------------------------------

class TestRowsToWide:

    def test_estructura_multiindex(self):
        rows = [
            {"date": "2026-09-15", "ticker": "BZ=F",
             "Open": 106.4, "High": 109.4, "Low": 105.4,
             "Close": 108.3, "Volume": 561034.0},
        ]
        df = _rows_to_wide(rows)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert isinstance(df.columns, pd.MultiIndex)
        assert ("Close", "BZ=F") in df.columns
        assert df.loc["2026-09-15", ("Close", "BZ=F")] == pytest.approx(108.3)

    def test_vacio(self):
        assert _rows_to_wide([]).empty

    def test_fecha_invalida_descartada(self):
        rows = [
            {"date": "no-fecha", "ticker": "BZ=F",
             "Open": 1.0, "High": 1.0, "Low": 1.0,
             "Close": 1.0, "Volume": 1.0},
        ]
        assert _rows_to_wide(rows).empty

    def test_dtypes_float64_con_nones(self):
        # K-FUTURES-DTYPE-01: rows con None en Open/High/Low/Volume (caso
        # spot real) deben salir como float64 tras _rows_to_wide, no object.
        rows = [
            {"date": "2026-09-15", "ticker": "GC=F",
             "Open": None, "High": None, "Low": None,
             "Close": 4341.77, "Volume": None},
        ]
        df = _rows_to_wide(rows)
        for field in ("Open", "High", "Low", "Close", "Volume"):
            assert (field, "GC=F") in df.columns, f"falta {field}"
            assert df[(field, "GC=F")].dtype == "float64", (
                f"{field} dtype={df[(field, 'GC=F')].dtype}, esperado float64"
            )
        # Valores None -> NaN
        assert pd.isna(df.loc["2026-09-15", ("Open", "GC=F")])
        assert pd.isna(df.loc["2026-09-15", ("Volume", "GC=F")])
        assert df.loc["2026-09-15", ("Close", "GC=F")] == pytest.approx(4341.77)

    def test_roundtrip_y_merge_sin_futurewarning(self, tmp_path):
        # K-FUTURES-DTYPE-01: prueba de la cadena real
        # _rows_to_wide -> to_parquet -> read_parquet -> merge.
        # Sin el cast en _rows_to_wide, el merge emite FutureWarning
        # (pandas 2.x) o falla (pandas 3.x).
        import warnings

        rows = [
            {"date": "2026-09-15", "ticker": "GC=F",
             "Open": None, "High": None, "Low": None,
             "Close": 4341.77, "Volume": None},
        ]
        wide = _rows_to_wide(rows)

        # Sanity: antes del cast no habria sido float64; ahora si.
        assert wide[("Open", "GC=F")].dtype == "float64"

        p = tmp_path / "spot.parquet"
        wide.to_parquet(p)

        # Releer el parquet: debe seguir float64 (caso real commodities_spot).
        check = pd.read_parquet(p)
        assert check[("Open", "GC=F")].dtype == "float64", (
            f"round-trip degrada a {check[('Open', 'GC=F')].dtype}"
        )

        # df_market con misma columna ya en float64 (target del merge)
        idx = pd.to_datetime(["2026-09-15"])
        cols = pd.MultiIndex.from_tuples(
            [("Open", "GC=F"), ("Close", "GC=F")], names=["field", "ticker"],
        )
        df_market = pd.DataFrame(0.0, index=idx, columns=cols, dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            out = merge_commodities_into_market(
                df_market, futures_path=None, spot_path=str(p),
            )

        assert out[("Open", "GC=F")].dtype == "float64"
        assert pd.isna(out.loc["2026-09-15", ("Open", "GC=F")])
        assert out.loc["2026-09-15", ("Close", "GC=F")] == pytest.approx(4341.77)



class TestToFloat:

    def test_string_a_float(self):
        assert _to_float("108.32") == pytest.approx(108.32)

    def test_none(self):
        assert _to_float(None) is None

    def test_invalido(self):
        assert _to_float("N/A") is None

    def test_int(self):
        assert _to_float(42) == 42.0


# --- Fetch (mockeado) --------------------------------------------

class TestFetchCommodities:

    def _mock_get(self, provider, mapping):
        def _fake(path, params=None):
            if path in mapping:
                return mapping[path]
            raise RuntimeError("endpoint no mockeado: " + path)
        provider._get = _fake

    def test_futures_y_spot(self):
        p = FuturesProvider()
        self._mock_get(p, {
            "/futures/ice-brent": BRENT_JSON,
            "/futures/ice-wti": WTI_JSON,
            "/prices/latest": SPOT_JSON,
        })
        df_fut, df_spot = p.fetch_commodities()

        assert not df_fut.empty
        assert not df_spot.empty
        assert ("Close", "BZ=F") in df_fut.columns
        assert ("Close", "CL=F") in df_fut.columns
        assert ("Close", "GC=F") in df_spot.columns
        assert ("Close", "HG=F") in df_spot.columns
        assert ("Close", "NG=F") in df_spot.columns
        assert df_fut.loc[pd.Timestamp("2026-09-15"), ("Close", "BZ=F")] == pytest.approx(108.32)

    def test_sin_settlement_date(self):
        p = FuturesProvider()
        bad = {"front_month": BRENT_JSON["front_month"]}  # sin settlement_date
        self._mock_get(p, {
            "/futures/ice-brent": bad,
            "/futures/ice-wti": bad,
            "/prices/latest": SPOT_JSON,
        })
        df_fut, _ = p.fetch_commodities()
        assert df_fut.empty


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
            {"Close": [999.0]},  # mismo 2026-09-15, debe ganar el nuevo
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
        # No debe explotar ni hacer pd.concat con DataFrame vacio
        assert out is df


# --- API key -----------------------------------------------------

class TestApiKey:

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("OIL_PRICE_API", "env_key_12345")
        p = FuturesProvider()
        assert p._get_api_key() == "env_key_12345"

    def test_fallback_local(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OIL_PRICE_API", raising=False)
        local = tmp_path / "key.txt"
        local.write_text("local_key_xyz\n", encoding="utf-8")
        monkeypatch.setattr(
            "data.providers.futures.LOCAL_KEY_FALLBACK", local
        )
        p = FuturesProvider()
        assert p._get_api_key() == "local_key_xyz"

    def test_sin_key_lanza(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OIL_PRICE_API", raising=False)
        monkeypatch.setattr(
            "data.providers.futures.LOCAL_KEY_FALLBACK",
            tmp_path / "no_existe.txt",
        )
        p = FuturesProvider()
        with pytest.raises(RuntimeError, match="OIL_PRICE_API"):
            p._get_api_key()


# --- Politica de reintentos --------------------------------------

class TestRetryPolicy:

    def test_401_no_reintenta(self, monkeypatch):
        class R:
            status_code = 401
            text = "no auth"

        calls = {"n": 0}
        def fake_get(*a, **kw):
            calls["n"] += 1
            return R()

        monkeypatch.setattr(
            "data.providers.futures.requests.Session.get", fake_get
        )
        monkeypatch.setenv("OIL_PRICE_API", "x")
        p = FuturesProvider()
        with pytest.raises(RuntimeError, match="401"):
            p._get("/prices/latest")
        assert calls["n"] == 1

    def test_429_no_reintenta(self, monkeypatch):
        class R:
            status_code = 429
            text = "rate limit"

        calls = {"n": 0}
        def fake_get(*a, **kw):
            calls["n"] += 1
            return R()

        monkeypatch.setattr(
            "data.providers.futures.requests.Session.get", fake_get
        )
        monkeypatch.setenv("OIL_PRICE_API", "x")
        p = FuturesProvider()
        with pytest.raises(RuntimeError, match="429"):
            p._get("/prices/latest")
        assert calls["n"] == 1
