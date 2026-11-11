# -*- coding: utf-8 -*-
"""Persistencia de la resolucion de contratos temporales (S35-bis).

Escribe outputs/state/temporal_contracts_result.json tras
resolve_all_contracts + build_temporal_meta. Objetivo: si un contrato
falla al resolver su fuente, este JSON es la unica evidencia persistente
de que contrato y por que. Complementa al gate 11/11, que ataja el
sintoma sin identificar el culpable.

Schema v1:
    {
      "schema_version": 1,
      "run_id": str,
      "reference_date": "YYYY-MM-DD",
      "global_last_date": "YYYY-MM-DD" | null,
      "generated_at_utc": "YYYY-MM-DDTHH:MM:SS+00:00",
      "contracts_total": int,
      "summary": {"OK": int, "STALE": int, "PENDING": int,
                  "INSUFFICIENT": int, "BLOCKED": int},
      "by_contract": {name: {status, effective_date, expected_date,
                             lag_days, coverage}}
    }

No usa output_registry: ese registro cubre outputs/history/*.csv. Los
JSONs de estado (mte_state.json, liquidity_state.json) se persisten
directos; este sigue el mismo patron.

Atomic write: tmp + os.replace para no dejar JSON a medias si el
proceso muere entre open() y flush().

generated_at_utc NO usa datetime.now() dentro de este modulo: recibe
`now` del caller o lo deriva de temporal_meta["reference_date"]. Esto
satisface el guard AST test_p7 (S31): la fecha de ejecucion se propaga
desde el origen, no se genera en cada funcion de src/.

TCR_STALE_HOURS: umbral que health_check usa para WARN si el JSON del
ultimo run es mas viejo. 48h justificado: el sistema tiene 5 slots
diarios; 48h sin persistencia exitosa es anomalia (no cubre huecos
normales entre slots).
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

from .base import (
    STATUS_BLOCKED,
    STATUS_INSUFFICIENT,
    STATUS_OK,
    STATUS_PENDING,
    STATUS_STALE,
)

TCR_SCHEMA_VERSION = 1
TCR_DEFAULT_PATH = "outputs/state/temporal_contracts_result.json"
TCR_STALE_HOURS = 48

_KNOWN_STATUSES = (
    STATUS_OK,
    STATUS_STALE,
    STATUS_PENDING,
    STATUS_INSUFFICIENT,
    STATUS_BLOCKED,
)


def _iso(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _as_utc_iso(now: datetime) -> str:
    """Convierte `now` a ISO string en UTC. Naive se interpreta como UTC."""
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc).isoformat()
    return now.astimezone(timezone.utc).isoformat()


def build_tcr_payload(temporal_meta: dict, now: datetime) -> dict:
    """Construye el dict schema v1 a partir de temporal_meta.

    `now` debe ser la fecha/hora de ejecucion (tz-aware en produccion,
    provista por run.py via temporal_meta["reference_date"] o por el
    caller directo). No usa datetime.now() para satisfacer test_p7.
    """
    if not isinstance(now, datetime):
        raise TypeError(f"now debe ser datetime, recibido {type(now).__name__}")

    by_contract = (temporal_meta or {}).get("by_contract") or {}

    normalized = {}
    for name, entry in by_contract.items():
        if not isinstance(entry, dict):
            normalized[name] = {
                "status": "UNKNOWN",
                "effective_date": None,
                "expected_date": None,
                "lag_days": None,
                "coverage": None,
            }
            continue
        normalized[name] = {
            "status": entry.get("status"),
            "effective_date": _iso(entry.get("effective_date")),
            "expected_date": _iso(entry.get("expected_date")),
            "lag_days": entry.get("lag_days"),
            "coverage": entry.get("coverage"),
        }

    summary = {status: 0 for status in _KNOWN_STATUSES}
    for entry in normalized.values():
        status = entry.get("status")
        if status in summary:
            summary[status] += 1

    ref_iso = _iso((temporal_meta or {}).get("reference_date"))
    if isinstance(ref_iso, str) and "T" in ref_iso:
        ref_iso = ref_iso.split("T", 1)[0]

    return {
        "schema_version": TCR_SCHEMA_VERSION,
        "run_id": (temporal_meta or {}).get("run_id") or "",
        "reference_date": ref_iso,
        "global_last_date": _iso((temporal_meta or {}).get("global_last_date")),
        "generated_at_utc": _as_utc_iso(now),
        "contracts_total": len(normalized),
        "summary": summary,
        "by_contract": normalized,
    }


def persist_temporal_contracts_result(
    temporal_meta: dict,
    output_path: str | os.PathLike | None = None,
    now: datetime | None = None,
) -> dict | None:
    """Persiste el JSON. Devuelve el payload o None si no-op.

    No-op cuando temporal_meta no trae by_contract. Evita un JSON con
    contracts_total=0 ambiguo ("pipeline no corrio" vs "corrio sin
    contratos").

    `now`: fecha/hora de ejecucion. Si None, se usa
    temporal_meta["reference_date"] (que build_temporal_meta setea
    siempre en produccion). Si tampoco esta, ValueError.
    """
    by_contract = (temporal_meta or {}).get("by_contract") or {}
    if not by_contract:
        return None

    if now is None:
        ref = (temporal_meta or {}).get("reference_date")
        if not isinstance(ref, datetime):
            raise ValueError(
                "persist_temporal_contracts_result requiere `now` o "
                "temporal_meta[\x27reference_date\x27] como datetime. "
                f"Recibido: reference_date={type(ref).__name__}"
            )
        now = ref

    payload = build_tcr_payload(temporal_meta, now)
    path = Path(output_path or TCR_DEFAULT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)
    return payload
