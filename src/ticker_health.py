"""Registro canonico de estado por ticker.

Fuente de verdad para exclusiones por evento corporativo (delisting,
fusion, adquisicion). Ningun consumidor debe hardcodear exclusiones;
todas viven en data/mappings/ticker_health.yaml y se consultan aqui.

Estados (ver yaml):
- ACTIVE: sin registro, elegible. Estado por defecto.
- DELISTED: evento corporativo confirmado. Excluido del universo.
- STALE: sin datos recientes, no confirmado. WARN, no excluido.
- TRUNCATED_IPO: IPO reciente con historia corta. No alerta.

Uso:
    from src.ticker_health import filter_eligible
    tickers = filter_eligible(get_stock_list())

Invariante: ningun consumidor de df_stocks debe operar sobre DELISTED.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import yaml


ESTADOS_VALIDOS = frozenset({"ACTIVE", "DELISTED", "STALE", "TRUNCATED_IPO"})
ESTADOS_EXCLUIDOS = frozenset({"DELISTED"})

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = _REPO_ROOT / "data" / "mappings" / "ticker_health.yaml"

_CACHE: Optional[dict] = None


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    tickers = raw.get("tickers") or {}
    if not isinstance(tickers, dict):
        raise ValueError(
            f"{path}: 'tickers' debe ser dict, es {type(tickers).__name__}"
        )
    for t, entry in tickers.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entrada {t} debe ser dict")
        estado = entry.get("estado")
        if estado not in ESTADOS_VALIDOS:
            raise ValueError(
                f"{path}: {t}: estado '{estado}' invalido. "
                f"Validos: {sorted(ESTADOS_VALIDOS)}"
            )
    return tickers


def _load_registry() -> dict:
    global _CACHE
    if _CACHE is None:
        _CACHE = _read_yaml(DEFAULT_PATH)
    return _CACHE


def _reset_cache() -> None:
    global _CACHE
    _CACHE = None


def get_estado(ticker: Optional[str]) -> str:
    """Devuelve el estado del ticker. ACTIVE si no esta registrado."""
    if not ticker:
        return "ACTIVE"
    entry = _load_registry().get(str(ticker).upper())
    if entry is None:
        return "ACTIVE"
    return entry["estado"]


def is_eligible(ticker: Optional[str]) -> bool:
    """True si el ticker debe operar en el universo (no DELISTED)."""
    return get_estado(ticker) not in ESTADOS_EXCLUIDOS


def filter_eligible(tickers: Iterable[str]) -> list:
    """Filtra tickers excluidos. Preserva orden, sin duplicados."""
    if tickers is None:
        return []
    seen = set()
    out = []
    for t in tickers:
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        if is_eligible(t):
            out.append(t)
    return out


def exclusion_reason(ticker: Optional[str]) -> Optional[str]:
    """Devuelve contexto + evidencia de exclusion, o None si elegible."""
    if not ticker:
        return None
    entry = _load_registry().get(str(ticker).upper())
    if entry is None:
        return None
    if entry.get("estado") not in ESTADOS_EXCLUIDOS:
        return None
    partes = []
    if entry.get("evento"):
        partes.append(entry["evento"])
    if entry.get("evidencia"):
        partes.append(entry["evidencia"])
    if not partes:
        partes.append(entry["estado"])
    return " | ".join(partes)