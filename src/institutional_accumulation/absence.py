"""P63 - Ausencia observada vs causa inferida.

Contrato semantico: docs/auditoria/iae/NIPC_CONTRATOS_SEMANTICOS_v1.md seccion 12.
Dictamen habilitante: A.6.2-bis-B3 (#48/#49).

Principio rector (P63): "una security no aparece en el filing publico"
NO implica "fue vendida".

Este modulo separa:
  - reporting_status   (hecho observado)
  - absence_reason     (causa inferida/conocida)
  - sale_evidence      (evidencia externa independiente)

Regla: la clasificacion automatica de absence_reason esta DIFERIDA v1
(seccion 12.4). Solo se exponen los enums y los helpers contractuales.
Cualquier intento de inferencia automatica falla explicitamente hasta
que exista evidencia suficiente y dictamen especifico.
"""
from __future__ import annotations


# --- reporting_status (hecho observado, seccion 12.2) ---
REPORTING_PRESENT = "PRESENT"
REPORTING_ZERO_REPORTED = "ZERO_REPORTED"
REPORTING_NOT_PRESENT = "NOT_PRESENT"

ALL_REPORTING_STATUSES = (
    REPORTING_PRESENT,
    REPORTING_ZERO_REPORTED,
    REPORTING_NOT_PRESENT,
)


# --- absence_reason (causa inferida/conocida, seccion 12.2) ---
REASON_MISSING = "MISSING"
REASON_BELOW_REPORTING_THRESHOLD = "BELOW_REPORTING_THRESHOLD"
REASON_CONFIDENTIAL = "CONFIDENTIAL"
REASON_OTHER_MANAGER = "OTHER_MANAGER"
REASON_UNKNOWN = "UNKNOWN"
ALL_ABSENCE_REASONS = (
    REASON_MISSING,
    REASON_BELOW_REPORTING_THRESHOLD,
    REASON_CONFIDENTIAL,
    REASON_OTHER_MANAGER,
    REASON_UNKNOWN,
)

DEFAULT_ABSENCE_REASON = REASON_MISSING


# --- sale_evidence (independiente, seccion 12.2) ---
SALE_NONE = "NONE"
SALE_DIRECT = "DIRECT"

ALL_SALE_EVIDENCE = (
    SALE_NONE,
    SALE_DIRECT,
)

DEFAULT_SALE_EVIDENCE = SALE_NONE


class AbsenceClassificationNotImplemented(NotImplementedError):
    """Inferencia automatica de absence_reason diferida (contrato seccion 12.4)."""


def classify_absence(*args, **kwargs):
    """Inferencia automatica de absence_reason.

    DIFERIDA v1. Requiere evidencia adicional y dictamen especifico
    (contrato seccion 12.4). Cualquier llamada falla explicitamente
    para evitar inferencias silenciosas no autorizadas.
    """
    raise AbsenceClassificationNotImplemented(
        "classify_absence esta diferida v1 (contrato seccion 12.4). "
        "No inferir ausencia -> venta sin evidencia externa."
    )