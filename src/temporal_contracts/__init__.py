"""API publica de src.temporal_contracts."""
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
    is_critical,
    get_contract,
    EquityEOD,
    IndexEODUSA,
    IndexEODEuropa,
    IndexEODCommodity,
    IndexEODCurrency,
    VolatilityIndex,
    RateYield,
    FutureSettlement,
    FxDailyCut,
    Sec13F,
    FinraDarkpool,
    SsgaPrimaryFlow,
    FredMacro,
    CftcPositionFlow,
    SecNport,
)
from src.temporal_contracts.validation import (
    validate_artifact_contracts,
    ValidationResult,
    ContractVerdict,
    MARKET_TO_CONTRACT,
    STATUS_OK as VALIDATION_OK,
    STATUS_WARN as VALIDATION_WARN,
    STATUS_BLOCKED as VALIDATION_BLOCKED,
    STATUS_UNAVAILABLE as VALIDATION_UNAVAILABLE,
)
from src.temporal_contracts.consolidate import (
    consolidate,
    build_temporal_meta,
)


def resolve_all_contracts(df_market, reference_date) -> dict:
    """Resuelve los contratos implementados. Devuelve dict[str, TemporalResolution]."""
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
    "is_critical",
    "validate_artifact_contracts",
    "ValidationResult",
    "ContractVerdict",
    "MARKET_TO_CONTRACT",
    "VALIDATION_OK",
    "VALIDATION_WARN",
    "VALIDATION_BLOCKED",
    "VALIDATION_UNAVAILABLE",
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
    "Sec13F",
    "FinraDarkpool",
    "SsgaPrimaryFlow",
    "FredMacro",
    "CftcPositionFlow",
    "SecNport",
]
