"""P61: modulo temporal_validity.

Encapsula operational_mapping_status (validez historica), separado
de security_resolution_status (identidad).
"""
from __future__ import annotations

import pandas as pd

STATUS_VERIFIED = "VERIFIED"
STATUS_TEMPORAL_UNVERIFIED = "TEMPORAL_UNVERIFIED"
STATUS_UNRESOLVED = "UNRESOLVED"
STATUS_CONFLICT = "CONFLICT"

ALL_STATUSES = (
    STATUS_VERIFIED,
    STATUS_TEMPORAL_UNVERIFIED,
    STATUS_UNRESOLVED,
    STATUS_CONFLICT,
)

SOURCES_WITH_VIGENCIA = frozenset({
    "cusip_equivalence",
    "cusip_ticker_exceptions",
})

SOURCES_WITHOUT_VIGENCIA = frozenset({
    "etf_holdings",
    "openfigi",
})


def _to_ts(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return pd.Timestamp(v).normalize()
    except Exception:
        return None


def resolve_source_status(source, valid_from, valid_to, period):
    """Estado de una fuente para un periodo."""
    p = _to_ts(period)
    if p is None:
        raise ValueError("period invalido: " + repr(period))

    if source in SOURCES_WITHOUT_VIGENCIA:
        return STATUS_TEMPORAL_UNVERIFIED

    if source in SOURCES_WITH_VIGENCIA:
        vf = _to_ts(valid_from)
        vt = _to_ts(valid_to)
        if vf is None:
            return STATUS_UNRESOLVED
        if vf <= p and (vt is None or vt >= p):
            return STATUS_VERIFIED
        return STATUS_UNRESOLVED

    return STATUS_UNRESOLVED


def aggregate_status(entries):
    """Agrega multiples (value, status) aplicando la regla Q7."""
    if not entries:
        return STATUS_UNRESOLVED

    verified = [(v, s) for v, s in entries if s == STATUS_VERIFIED]
    temp_unv = [(v, s) for v, s in entries if s == STATUS_TEMPORAL_UNVERIFIED]

    if verified:
        vals = {v for v, _ in verified}
        if len(vals) > 1:
            return STATUS_CONFLICT
        v_ref = next(iter(vals))
        for v, _ in temp_unv:
            if v != v_ref:
                return STATUS_CONFLICT
        return STATUS_VERIFIED

    if temp_unv:
        vals = {v for v, _ in temp_unv}
        if len(vals) > 1:
            return STATUS_CONFLICT
        return STATUS_TEMPORAL_UNVERIFIED

    return STATUS_UNRESOLVED