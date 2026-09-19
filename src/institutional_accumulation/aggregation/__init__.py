"""Paquete aggregation - NIPC (Net Institutional Position Change).

Dictamen habilitante: docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_
ESPECIFICACION.md (v1.2) seccion 11.1.

Modulos:
  - delta_shares.py    compute_reported_position_units + compute_delta_shares
  - nipc.py            compute_nipc + compute_coverage_pairwise (siguiente)

Contrato de pureza (spec 11.3):
  - NO leen ficheros (reciben DataFrames).
  - NO escriben ficheros.
  - NO usan datetime.now() como fecha de observacion.
  - Deterministas y testeables.

La orquestacion (leer parquets, escribir resultado) vive en scripts/
o en un orchestrator separado.

Fuera de aggregation/:
  - identity/  resuelve quien / que / cuando
  - pipeline/  orquesta el run diario
"""
from __future__ import annotations

from .delta_shares import (
    ALL_MATCH_STATUSES,
    DELTA_COLUMNS,
    MATCH_KEY,
    STATUS_BOTH,
    STATUS_EXIT,
    STATUS_NEW,
    STATUS_UNRESOLVED_IDENTITY,
    UNITS_COLUMNS,
    VALID_DISCRETIONS,
    compute_delta_shares,
    compute_reported_position_units,
)

__all__ = [
    "compute_reported_position_units",
    "compute_delta_shares",
    "MATCH_KEY",
    "VALID_DISCRETIONS",
    "STATUS_BOTH",
    "STATUS_NEW",
    "STATUS_EXIT",
    "STATUS_UNRESOLVED_IDENTITY",
    "ALL_MATCH_STATUSES",
    "UNITS_COLUMNS",
    "DELTA_COLUMNS",
]