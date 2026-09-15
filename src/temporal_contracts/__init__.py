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


_IMPLEMENTED_CONTRACTS = {
    "EQUITY_EOD": EquityEOD,
    "INDEX_EOD_USA": IndexEODUSA,
    "INDEX_EOD_EUROPA": IndexEODEuropa,
    "INDEX_EOD_COMMODITY": IndexEODCommodity,
    "INDEX_EOD_CURRENCY": IndexEODCurrency,
}


def get_contract(name: str) -> TemporalContract:
    """Devuelve una nueva instancia del contrato name.

    Lanza KeyError si el contrato no tiene implementacion en esta fase.
    Fase 2.1: EQUITY_EOD + los 4 INDEX_EOD_*.
    """
    if name not in _IMPLEMENTED_CONTRACTS:
        raise KeyError(
            "Contrato no implementado en esta fase: "
            + name
            + ". Implementados: "
            + str(sorted(_IMPLEMENTED_CONTRACTS))
        )
    return _IMPLEMENTED_CONTRACTS[name]()


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
    "EquityEOD",
    "IndexEODUSA",
    "IndexEODEuropa",
    "IndexEODCommodity",
    "IndexEODCurrency",
]