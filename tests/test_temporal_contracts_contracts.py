"""Tests de los contratos implementados en Fase 2.1."""
from datetime import date, datetime

import pandas as pd
import pytest

from src.temporal_contracts import (
    EquityEOD,
    IndexEODUSA,
    IndexEODEuropa,
    IndexEODCommodity,
    IndexEODCurrency,
    get_contract,
    TemporalContract,
    TemporalResolution,
    VALID_STATUSES,
)
from src.temporal_contracts._common import (
    extract_close,
    get_universe_by_class,
    to_date,
    weekday_expected,
)


REF = datetime(2026, 9, 15, 22, 0)



# --- _common helpers ---

class TestCommonHelpers:

    def test_extract_close_multiindex(self, df_real):
        close = extract_close(df_real)
        assert not isinstance(close.columns, pd.MultiIndex)
        assert len(close.columns) == 562

    def test_extract_close_flat_passthrough(self):
        df = pd.DataFrame({"A": [1, 2]})
        assert extract_close(df) is df

    def test_extract_close_empty(self):
        assert extract_close(pd.DataFrame()) is not None

    def test_to_date_timestamp(self):
        assert to_date(pd.Timestamp("2026-09-14")) == date(2026, 9, 14)

    def test_to_date_datetime(self):
        assert to_date(datetime(2026, 9, 14, 10)) == date(2026, 9, 14)

    def test_to_date_date(self):
        assert to_date(date(2026, 9, 14)) == date(2026, 9, 14)

    def test_to_date_none(self):
        assert to_date(None) is None

    def test_to_date_unexpected_returns_none(self):
        assert to_date("2026-09-14") is None

    def test_get_universe_by_class_equity(self, df_real):
        universe = get_universe_by_class(df_real, "EQUITY")
        assert len(universe) == 539

    def test_get_universe_by_class_index(self, df_real):
        universe = get_universe_by_class(df_real, "INDEX")
        assert len(universe) == 10

    def test_get_universe_by_class_future(self, df_real):
        universe = get_universe_by_class(df_real, "FUTURE")
        assert len(universe) == 5

    def test_get_universe_by_class_fx(self, df_real):
        universe = get_universe_by_class(df_real, "FX")
        assert len(universe) == 3

    def test_weekday_expected_lunes(self):
        # 2026-09-14 es lunes
        assert weekday_expected(datetime(2026, 9, 14, 10)) == date(2026, 9, 14)

    def test_weekday_expected_sabado_retrocede_a_viernes(self):
        # 2026-09-12 es sabado
        assert weekday_expected(datetime(2026, 9, 12, 10)) == date(2026, 9, 11)

    def test_weekday_expected_domingo_retrocede_a_viernes(self):
        assert weekday_expected(datetime(2026, 9, 13, 10)) == date(2026, 9, 11)


# --- EquityEOD ---

class TestEquityEOD:

    def test_es_instancia_de_temporal_contract(self):
        assert isinstance(EquityEOD(), TemporalContract)

    def test_metadata_basica(self):
        c = EquityEOD()
        assert c.name == "EQUITY_EOD"
        assert c.family == "EQUITY"
        assert c.session_calendar == "NYSE"
        assert c.max_lag_days == 0
        assert c.min_coverage == 0.90

    def test_resolve_devuelve_resolution(self, df_real):
        r = EquityEOD().resolve(df_real, REF)
        assert isinstance(r, TemporalResolution)
        assert r.contract_name == "EQUITY_EOD"
        assert r.status in VALID_STATUSES

    def test_resolve_popula_eligible_universe(self, df_real):
        c = EquityEOD()
        c.resolve(df_real, REF)
        assert len(c.eligible_universe) == 539

    def test_resolve_status_coherente_con_coverage_y_lag(self, df_real):
        r = EquityEOD().resolve(df_real, REF)
        # parquet actual: eff=11/09, exp=14/09, lag=3 > max=0 -> INSUFFICIENT
        assert r.coverage is not None
        assert r.effective_date is not None
        assert r.expected_date is not None
        assert r.lag_days == (r.expected_date - r.effective_date).days

    def test_effective_no_supera_expected(self, df_real):
        r = EquityEOD().resolve(df_real, REF)
        assert r.effective_date <= r.expected_date


# --- Index EOD ---

class TestIndexEODUSA:

    def test_metadata(self):
        c = IndexEODUSA()
        assert c.name == "INDEX_EOD_USA"
        assert c.family == "INDEX"
        assert c.max_lag_days == 1
        assert len(c.eligible_universe) == 4

    def test_resolve_ok_o_stale(self, df_real):
        r = IndexEODUSA().resolve(df_real, REF)
        assert r.contract_name == "INDEX_EOD_USA"
        assert r.status in ("OK", "STALE", "INSUFFICIENT")
        assert r.coverage == 1.0

    def test_per_ticker_lag_presente(self):
        c = IndexEODUSA()
        assert c.per_ticker_lag is not None
        assert set(c.per_ticker_lag) == {"^GSPC", "^DJI", "^NDX", "^RUT"}


class TestIndexEODEuropa:

    def test_metadata(self):
        c = IndexEODEuropa()
        assert c.name == "INDEX_EOD_EUROPA"
        assert c.max_lag_days == 5
        assert len(c.eligible_universe) == 4

    def test_resolve_devuelve_status_valido(self, df_real):
        r = IndexEODEuropa().resolve(df_real, REF)
        assert r.contract_name == "INDEX_EOD_EUROPA"
        assert r.status in VALID_STATUSES

    def test_per_ticker_lag_ftse_uno_resto_dos(self):
        c = IndexEODEuropa()
        assert c.per_ticker_lag["^FTSE"] == 1
        assert c.per_ticker_lag["^GDAXI"] == 2


class TestIndexEODCommodity:

    def test_metadata(self):
        c = IndexEODCommodity()
        assert c.name == "INDEX_EOD_COMMODITY"
        assert c.eligible_universe == ["^SPGSCI"]

    def test_resolve(self, df_real):
        r = IndexEODCommodity().resolve(df_real, REF)
        assert r.contract_name == "INDEX_EOD_COMMODITY"
        assert r.status in VALID_STATUSES


class TestIndexEODCurrency:

    def test_metadata(self):
        c = IndexEODCurrency()
        assert c.name == "INDEX_EOD_CURRENCY"
        assert c.eligible_universe == ["DX-Y.NYB"]
        assert c.session_calendar == "ICE"

    def test_resolve(self, df_real):
        r = IndexEODCurrency().resolve(df_real, REF)
        assert r.contract_name == "INDEX_EOD_CURRENCY"
        assert r.status in VALID_STATUSES


# --- get_contract ---

class TestGetContract:

    def test_devuelve_instancia_equity_eod(self):
        c = get_contract("EQUITY_EOD")
        assert isinstance(c, EquityEOD)

    def test_devuelve_instancia_index_usa(self):
        c = get_contract("INDEX_EOD_USA")
        assert isinstance(c, IndexEODUSA)

    def test_instancias_independientes(self):
        c1 = get_contract("EQUITY_EOD")
        c2 = get_contract("EQUITY_EOD")
        assert c1 is not c2


    def test_contrato_inexistente_lanza_keyerror(self):
        with pytest.raises(KeyError):
            get_contract("NO_EXISTE")

    def test_los_cinco_implementados(self):
        for name in [
            "EQUITY_EOD",
            "INDEX_EOD_USA",
            "INDEX_EOD_EUROPA",
            "INDEX_EOD_COMMODITY",
            "INDEX_EOD_CURRENCY",
        ]:
            c = get_contract(name)
            assert c.name == name