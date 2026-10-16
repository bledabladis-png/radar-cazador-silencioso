"""Catalogo declarativo de los 9 contratos temporales (FU-021-5 Fase 1).

Fase 1 solo declara el catalogo. Las clases con resolve() se anaden en Fase 2.

Enmienda v1.1 (2026-10-16): anadido campo `critical` por contrato.
Los contratos marcados critical=True gobiernan el bloqueo de la
finalizacion del pipeline.

Fuente: docs/auditoria/FU-021-5_ESPECIFICACION_CONTRATO_TEMPORAL.md (Parte A v2)
        docs/auditoria/FU-021-5_ESPECIFICACION_CONTRATO_TEMPORAL_PARTE_B.md
        docs/auditoria/daily_run_gate_enmienda_v1_1.md
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
        "critical": True,
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
        "critical": False,
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
        "critical": False,
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
        "critical": False,
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
        "critical": False,
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
        "critical": False,
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
        "critical": False,
    },
    "FUTURE_SETTLEMENT": {
        "family": "FUTURE",
        "eligible_universe": ["BZ=F", "CL=F", "GC=F", "HG=F", "NG=F"],
        "session_calendar": "CME|NYMEX|COMEX",
        "max_lag_days": 1,
        "min_coverage": 1.0,
        "per_ticker_lag": {"BZ=F": 1, "CL=F": 1, "GC=F": 1, "HG=F": 1, "NG=F": 1},
        "per_pair_max_lag": None,
        "activation_req": None,
        "critical": False,
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
        "critical": False,
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


def is_critical(name: str) -> bool:
    """True si el contrato es obligatorio (bloquea la finalizacion)."""
    return bool(get_registry_entry(name).get("critical", False))