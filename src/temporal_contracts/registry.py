"""Catalogo declarativo de los 9 contratos temporales (FU-021-5 Fase 1).

Fase 1 solo declara el catalogo. Las clases con resolve() se anaden en Fase 2.

Fuente: docs/auditoria/FU-021-5_ESPECIFICACION_CONTRATO_TEMPORAL.md (Parte A v2)
        docs/auditoria/FU-021-5_ESPECIFICACION_CONTRATO_TEMPORAL_PARTE_B.md
"""
from __future__ import annotations


CONTRACTS_REGISTRY = {
    "EQUITY_EOD": {
        "family": "EQUITY",
        "eligible_universe_count": 539,
        "session_calendar": "NYSE",
        "max_lag_days": 0,
        "min_coverage": 0.90,
        "per_ticker_lag": None,
        "per_pair_max_lag": None,
        "activation_req": None,
    },
    "INDEX_EOD_USA": {
        "family": "INDEX",
        "eligible_universe": ["^GSPC", "^DJI", "^NDX", "^RUT"],
        "session_calendar": "NYSE",
        "max_lag_days": 1,
        "min_coverage": 1.0,
        "per_ticker_lag": {"^GSPC": 1, "^DJI": 1, "^NDX": 1, "^RUT": 1},
        "per_pair_max_lag": None,
        "activation_req": None,
    },
    "INDEX_EOD_EUROPA": {
        "family": "INDEX",
        "eligible_universe": ["^FTSE", "^GDAXI", "^IBEX", "^STOXX50E"],
        "session_calendar": "LSE|XETRA|BME|EURONEXT",
        "max_lag_days": 5,
        "min_coverage": 1.0,
        "per_ticker_lag": {"^FTSE": 1, "^GDAXI": 2, "^IBEX": 2, "^STOXX50E": 2},
        "per_pair_max_lag": None,
        "activation_req": None,
    },
    "INDEX_EOD_COMMODITY": {
        "family": "INDEX",
        "eligible_universe": ["^SPGSCI"],
        "session_calendar": "NYSE",
        "max_lag_days": 1,
        "min_coverage": 1.0,
        "per_ticker_lag": {"^SPGSCI": 1},
        "per_pair_max_lag": None,
        "activation_req": None,
    },
    "INDEX_EOD_CURRENCY": {
        "family": "INDEX",
        "eligible_universe": ["DX-Y.NYB"],
        "session_calendar": "ICE",
        "max_lag_days": 1,
        "min_coverage": 1.0,
        "per_ticker_lag": {"DX-Y.NYB": 1},
        "per_pair_max_lag": None,
        "activation_req": None,
    },
    "VOLATILITY_INDEX": {
        "family": "INDEX",
        "eligible_universe": ["^VIX", "^VIX3M", "^VXN"],
        "session_calendar": "NYSE",
        "max_lag_days": 1,
        "min_coverage": 1.0,
        "per_ticker_lag": {"^VIX": 1, "^VIX3M": 1, "^VXN": 1},
        "per_pair_max_lag": None,
        "activation_req": None,
    },
    "RATE_YIELD": {
        "family": "RATE_YIELD",
        "eligible_universe": ["^FVX", "^TNX"],
        "session_calendar": "NYSE",
        "max_lag_days": 1,
        "min_coverage": 1.0,
        "per_ticker_lag": {"^FVX": 1, "^TNX": 1},
        "per_pair_max_lag": None,
        "activation_req": None,
    },
    "FUTURE_SETTLEMENT": {
        "family": "FUTURE",
        "eligible_universe": ["BZ=F", "CL=F", "GC=F", "HG=F", "NG=F"],
        "session_calendar": "CME|ICE",
        "max_lag_days": None,
        "min_coverage": None,
        "per_ticker_lag": None,
        "per_pair_max_lag": None,
        "activation_req": {
            "provider_specific_contract": True,
            "official_settlement": True,
            "stable_identifier": True,
            "declared_rollover": True,
        },
    },
    "SPOT_COMMODITY": {
        "family": "SPOT",
        "eligible_universe": ["GC=F", "HG=F", "NG=F"],
        "session_calendar": "COMEX|NYMEX",
        "max_lag_days": 1,
        "min_coverage": 1.0,
        "per_ticker_lag": {"GC=F": 1, "HG=F": 1, "NG=F": 1},
        "per_pair_max_lag": None,
        "activation_req": None,
    },
    "FX_DAILY_CUT": {
        "family": "FX",
        "eligible_universe": ["EURUSD=X", "USDCNY=X", "USDJPY=X"],
        "session_calendar": "FX_24h",
        "max_lag_days": {"EURUSD=X": 0, "USDJPY=X": 0, "USDCNY=X": 1},
        "min_coverage": 0.5,
        "per_ticker_lag": None,
        "per_pair_max_lag": {"EURUSD=X": 0, "USDJPY=X": 0, "USDCNY=X": 1},
        "activation_req": None,
    },
}


def list_contracts() -> list:
    """Devuelve los nombres de los 9 contratos registrados."""
    return list(CONTRACTS_REGISTRY.keys())


def get_registry_entry(name: str) -> dict:
    """Devuelve la entrada del catalogo para un contrato."""
    if name not in CONTRACTS_REGISTRY:
        raise KeyError(f"Contrato no registrado: {name}")
    return CONTRACTS_REGISTRY[name]