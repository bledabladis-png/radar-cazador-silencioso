"""Tests de src/temporal_contracts/registry.py (FU-021-5 Fase 1 + S28)."""

from src.temporal_contracts.registry import (
    _IMPLEMENTED_CONTRACTS,
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
    "FX_DAILY_CUT",
    "SEC_13F",
    "FINRA_DARKPOOL",
    "SSGA_PRIMARY_FLOW",
    "FRED_MACRO",
    "CFTC_POSITION_FLOW",
    "SEC_NPORT",
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
    "FX_DAILY_CUT": "FX",
    "SEC_13F": "SEC",
    "FINRA_DARKPOOL": "FINRA",
    "SSGA_PRIMARY_FLOW": "SSGA",
    "FRED_MACRO": "FRED",
    "CFTC_POSITION_FLOW": "CFTC",
    "SEC_NPORT": "SEC",
}


class TestRegistryShape:

    def test_trece_contratos(self):
        assert len(CONTRACTS_REGISTRY) == 15

    def test_nombres_exactos(self):
        assert list(CONTRACTS_REGISTRY.keys()) == EXPECTED_CONTRACTS

    def test_list_contracts_devuelve_los_nombres(self):
        assert list_contracts() == EXPECTED_CONTRACTS

    def test_familias_correctas(self):
        for name, expected_family in EXPECTED_FAMILIES.items():
            assert CONTRACTS_REGISTRY[name]["family"] == expected_family

    def test_nueve_familias_distintas(self):
        families = {v["family"] for v in CONTRACTS_REGISTRY.values()}
        assert families == {
            "EQUITY", "INDEX", "RATE_YIELD", "FUTURE", "FX",
            "SEC", "FINRA", "SSGA", "FRED", "CFTC",
        }

    # --- S28: contrato de derivacion (drift guard) ---

    def test_registry_deriva_de_implemented_contracts(self):
        """Si anades una clase a _IMPLEMENTED_CONTRACTS y no aparece en
        CONTRACTS_REGISTRY, este test falla. Cierra la clase de bug del
        S28: 4 contratos registrados pero no cableados."""
        assert set(CONTRACTS_REGISTRY.keys()) == set(_IMPLEMENTED_CONTRACTS.keys())

    def test_derivacion_por_clase(self):
        """Los metadatos del catalogo provienen de la clase (fuente unica)."""
        for name, cls in _IMPLEMENTED_CONTRACTS.items():
            entry = CONTRACTS_REGISTRY[name]
            assert entry["family"] == cls.family, name
            assert entry["session_calendar"] == cls.session_calendar, name
            assert entry["max_lag_days"] == cls.max_lag_days, name
            assert entry["min_coverage"] == cls.min_coverage, name
            assert entry["per_ticker_lag"] == cls.per_ticker_lag, name
            assert entry["per_pair_max_lag"] == cls.per_pair_max_lag, name
            assert entry["activation_req"] == cls.activation_req, name

    def test_critical_solo_equity_eod(self):
        """Solo EQUITY_EOD bloquea la publicacion (enmienda v1.1)."""
        critical_contracts = [
            name for name, e in CONTRACTS_REGISTRY.items() if e.get("critical")
        ]
        assert critical_contracts == ["EQUITY_EOD"]

    def test_cuatro_contratos_s28_presentes(self):
        """Los 4 contratos anadidos en S28 estan cableados."""
        s28 = {"SEC_13F", "FINRA_DARKPOOL", "SSGA_PRIMARY_FLOW", "FRED_MACRO"}
        assert s28.issubset(CONTRACTS_REGISTRY.keys())


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
        assert set(e["eligible_universe"]) == {
            "BZ=F", "CL=F", "GC=F", "HG=F", "NG=F"
        }

    def test_fx_daily_cut_tiene_per_pair_max_lag(self):
        e = get_registry_entry("FX_DAILY_CUT")
        assert e["per_pair_max_lag"] is not None
