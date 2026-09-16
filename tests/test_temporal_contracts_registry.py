"""Tests de src/temporal_contracts/registry.py (FU-021-5 Fase 1)."""
import pytest

from src.temporal_contracts.registry import (
    CONTRACTS_REGISTRY,
    list_contracts,
    get_registry_entry,
)


EXPECTED_CONTRACTS = [
    "EQUITY_EOD",
    "INDEX_EOD_USA",
    "INDEX_EOD_EUROPA",
    "INDEX_EOD_COMMODITY",
    "INDEX_EOD_CURRENCY",
    "VOLATILITY_INDEX",
    "RATE_YIELD",
    "FUTURE_SETTLEMENT",
    "SPOT_COMMODITY",
    "FX_DAILY_CUT",
]

EXPECTED_FAMILIES = {
    "EQUITY_EOD": "EQUITY",
    "INDEX_EOD_USA": "INDEX",
    "INDEX_EOD_EUROPA": "INDEX",
    "INDEX_EOD_COMMODITY": "INDEX",
    "INDEX_EOD_CURRENCY": "INDEX",
    "VOLATILITY_INDEX": "INDEX",
    "RATE_YIELD": "RATE_YIELD",
    "FUTURE_SETTLEMENT": "FUTURE",
    "SPOT_COMMODITY": "SPOT",
    "FX_DAILY_CUT": "FX",
}


class TestRegistryShape:

    def test_diez_contratos(self):
        assert len(CONTRACTS_REGISTRY) == 10

    def test_nombres_exactos(self):
        assert list(CONTRACTS_REGISTRY.keys()) == EXPECTED_CONTRACTS

    def test_list_contracts_devuelve_los_nombres(self):
        assert list_contracts() == EXPECTED_CONTRACTS

    def test_familias_correctas(self):
        for name, expected_family in EXPECTED_FAMILIES.items():
            assert CONTRACTS_REGISTRY[name]["family"] == expected_family

    def test_seis_familias_distintas(self):
        families = {v["family"] for v in CONTRACTS_REGISTRY.values()}
        assert families == {"EQUITY", "INDEX", "RATE_YIELD", "FUTURE", "FX", "SPOT"}


class TestRegistryEntries:

    def test_equity_eod_usa_nyse_min_coverage_alto(self):
        e = get_registry_entry("EQUITY_EOD")
        assert e["session_calendar"] == "NYSE"
        assert e["min_coverage"] == 0.90
        assert e["max_lag_days"] == 0

    def test_index_eod_europa_tiene_per_ticker_lag(self):
        e = get_registry_entry("INDEX_EOD_EUROPA")
        assert e["per_ticker_lag"] is not None
        assert "^FTSE" in e["per_ticker_lag"]

    def test_future_settlement_activo(self):
        e = get_registry_entry("FUTURE_SETTLEMENT")
        assert e["activation_req"] is None
        assert e["max_lag_days"] == 1
        assert e["min_coverage"] == 1.0
        assert set(e["eligible_universe"]) == {"BZ=F", "CL=F"}

    def test_fx_daily_cut_tiene_per_pair_max_lag(self):
        e = get_registry_entry("FX_DAILY_CUT")
        assert e["per_pair_max_lag"] is not None
        assert e["per_pair_max_lag"]["USDCNY=X"] == 1
        assert e["per_pair_max_lag"]["EURUSD=X"] == 0

    def test_index_eod_currency_es_dx_y_nyb(self):
        e = get_registry_entry("INDEX_EOD_CURRENCY")
        assert e["eligible_universe"] == ["DX-Y.NYB"]
        assert e["session_calendar"] == "ICE"

    def test_index_eod_commodity_es_spgsci(self):
        e = get_registry_entry("INDEX_EOD_COMMODITY")
        assert e["eligible_universe"] == ["^SPGSCI"]

    def test_contrato_no_registrado_lanza_keyerror(self):
        with pytest.raises(KeyError):
            get_registry_entry("NO_EXISTE")

    def test_volatility_index_usa_nyse(self):
        e = get_registry_entry("VOLATILITY_INDEX")
        assert e["session_calendar"] == "NYSE"
        assert set(e["eligible_universe"]) == {"^VIX", "^VIX3M", "^VXN"}