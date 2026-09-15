"""API publica de src/temporal_contracts."""
from src.temporal_contracts.base import (
    TemporalContract,
    TemporalResolution,
    MarketDataBundle,
    compute_status,
    STATUS_PENDING,
    STATUS_OK,
    STATUS_STALE,
    STATUS_INSUFFICIENT,
    STATUS_BLOCKED,
    VALID_STATUSES,
)
from src.temporal_contracts.registry import (
    CONTRACTS_REGISTRY,
    list_contracts,
    get_registry_entry,
)
from src.temporal_contracts.equity_eod import EquityEOD
from src.temporal_contracts.index_eod import (
    IndexEODUSA,
    IndexEODEuropa,
    IndexEODCommodity,
    IndexEODCurrency,
)
from src.temporal_contracts.volatility_index import VolatilityIndex
from src.temporal_contracts.rate_yield import RateYield
from src.temporal_contracts.future_settlement import FutureSettlement
from src.temporal_contracts.fx_daily_cut import FxDailyCut
from src.temporal_contracts.consolidate import (
    consolidate,
    build_temporal_meta,
)


_IMPLEMENTED_CONTRACTS = {
    "EQUITY_EOD": EquityEOD,
    "INDEX_EOD_USA": IndexEODUSA,
    "INDEX_EOD_EUROPA": IndexEODEuropa,
    "INDEX_EOD_COMMODITY": IndexEODCommodity,
    "INDEX_EOD_CURRENCY": IndexEODCurrency,
    "VOLATILITY_INDEX": VolatilityIndex,
    "RATE_YIELD": RateYield,
    "FUTURE_SETTLEMENT": FutureSettlement,
    "FX_DAILY_CUT": FxDailyCut,
}


def get_contract(name: str) -> TemporalContract:
    """Devuelve una nueva instancia del contrato name."""
    if name not in _IMPLEMENTED_CONTRACTS:
        raise KeyError(
            "Contrato no registrado: " + name
            + ". Implementados: " + str(sorted(_IMPLEMENTED_CONTRACTS))
        )
    return _IMPLEMENTED_CONTRACTS[name]()


def resolve_all_contracts(df_market, reference_date) -> dict:
    """Resuelve los 9 contratos. Devuelve dict[str, TemporalResolution]."""
    result = {}
    for name in list_contracts():
        result[name] = get_contract(name).resolve(df_market, reference_date)
    return result


__all__ = [
    "TemporalContract",
    "TemporalResolution",
    "MarketDataBundle",
    "compute_status",
    "STATUS_PENDING",
    "STATUS_OK",
    "STATUS_STALE",
    "STATUS_INSUFFICIENT",
    "STATUS_BLOCKED",
    "VALID_STATUSES",
    "CONTRACTS_REGISTRY",
    "list_contracts",
    "get_registry_entry",
    "get_contract",
    "resolve_all_contracts",
    "consolidate",
    "build_temporal_meta",
    "EquityEOD",
    "IndexEODUSA",
    "IndexEODEuropa",
    "IndexEODCommodity",
    "IndexEODCurrency",
    "VolatilityIndex",
    "RateYield",
    "FutureSettlement",
    "FxDailyCut",
]