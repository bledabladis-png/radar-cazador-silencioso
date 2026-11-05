"""Catalogo declarativo de los contratos temporales (FU-021-5 Fase 1).

S28 (2026-11-04): el catalogo se DERIVA de `_IMPLEMENTED_CONTRACTS` y de
los atributos de clase. Antes de S28 habia dos fuentes de verdad
(CONTRACTS_REGISTRY con 9 entradas literales y _IMPLEMENTED_CONTRACTS
con 13); los 4 contratos anadidos en S28 quedaban registrados pero no
resueltos en produccion. El fix elimina la lista paralela: anadir un
contrato = anadir la clase + registrarla en `_IMPLEMENTED_CONTRACTS`.

Enmienda v1.1 (2026-10-16): campo `critical` por contrato. Solo
EQUITY_EOD es critical=True.

Nota: `EQUITY_EOD` declara su universo dinamicamente (`self.eligible_universe`
en `__init__` via `get_universe_by_class`). El catalogo omite
`eligible_universe` para ese contrato. Los 4 contratos anadidos en S28
tambien lo omiten (universo no es lista de tickers). El campo legacy
`eligible_universe_count` se retira: 0 consumidores verificados en S28.

Fuente: docs/auditoria/FU-021-5_ESPECIFICACION_CONTRATO_TEMPORAL.md (Parte A v2)
        docs/auditoria/FU-021-5_ESPECIFICACION_CONTRATO_TEMPORAL_PARTE_B.md
        docs/auditoria/daily_run_gate_enmienda_v1_1.md
"""
from __future__ import annotations

from src.temporal_contracts.base import TemporalContract
from src.temporal_contracts.equity_eod import EquityEOD
from src.temporal_contracts.finra_darkpool import FinraDarkpool
from src.temporal_contracts.fred_macro import FredMacro
from src.temporal_contracts.future_settlement import FutureSettlement
from src.temporal_contracts.fx_daily_cut import FxDailyCut
from src.temporal_contracts.index_eod import (
    IndexEODCommodity,
    IndexEODCurrency,
    IndexEODEuropa,
    IndexEODUSA,
)
from src.temporal_contracts.rate_yield import RateYield
from src.temporal_contracts.sec_13f import Sec13F
from src.temporal_contracts.ssga_primary_flow import SsgaPrimaryFlow
from src.temporal_contracts.volatility_index import VolatilityIndex


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
    "SEC_13F": Sec13F,
    "FINRA_DARKPOOL": FinraDarkpool,
    "SSGA_PRIMARY_FLOW": SsgaPrimaryFlow,
    "FRED_MACRO": FredMacro,
}


def _build_registry_entry(cls: type) -> dict:
    """Deriva la entrada del catalogo desde los atributos de clase.

    Fuente unica de verdad: la clase. El registro es una vista derivada.
    `eligible_universe` se omite cuando la clase no lo declara como
    atributo (universo dinamico via `__init__` o lista vacia).
    """
    entry = {
        "family": cls.family,
        "session_calendar": cls.session_calendar,
        "max_lag_days": cls.max_lag_days,
        "min_coverage": cls.min_coverage,
        "per_ticker_lag": cls.per_ticker_lag,
        "per_pair_max_lag": cls.per_pair_max_lag,
        "activation_req": cls.activation_req,
        "critical": cls.critical,
    }
    _eu = getattr(cls, "eligible_universe", None)
    if _eu:
        entry["eligible_universe"] = _eu
    return entry


CONTRACTS_REGISTRY = {
    name: _build_registry_entry(cls)
    for name, cls in _IMPLEMENTED_CONTRACTS.items()
}


def list_contracts() -> list:
    """Devuelve los nombres de los contratos implementados."""
    return list(_IMPLEMENTED_CONTRACTS.keys())


def get_registry_entry(name: str) -> dict:
    """Devuelve la entrada del catalogo para un contrato."""
    if name not in CONTRACTS_REGISTRY:
        raise KeyError(f"Contrato no registrado: {name}")
    return CONTRACTS_REGISTRY[name]


def is_critical(name: str) -> bool:
    """True si el contrato es obligatorio (bloquea la finalizacion)."""
    return bool(get_registry_entry(name).get("critical", False))


def get_contract(name: str) -> TemporalContract:
    """Devuelve una nueva instancia del contrato name."""
    if name not in _IMPLEMENTED_CONTRACTS:
        raise KeyError(
            "Contrato no registrado: " + name
            + ". Implementados: " + str(sorted(_IMPLEMENTED_CONTRACTS))
        )
    return _IMPLEMENTED_CONTRACTS[name]()
