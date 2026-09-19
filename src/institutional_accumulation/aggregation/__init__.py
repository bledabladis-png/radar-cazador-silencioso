"""Paquete aggregation - NIPC (Net Institutional Position Change).

Dictamen habilitante: docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_
ESPECIFICACION.md (v1.2) seccion 11.1.

Modulos previstos:
  - delta_shares.py    compute_reported_position_units + compute_delta_shares
  - nipc.py            compute_nipc + compute_coverage_pairwise

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

__all__ = []  # se amplia al implementar delta_shares y nipc