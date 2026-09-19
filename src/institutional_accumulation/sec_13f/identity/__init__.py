"""Modulo identity - capa de identidad del framework 3 niveles (FA-2).

FA-2.1: temporal_filter.py (filtro canonico por PERIODOFREPORT).
FA-2.2: cusip_resolver.py (CUSIP -> ticker con vigencia temporal).
FA-2.3: relationships.py (Column 7 -> OTHERMANAGER2.SEQUENCENUMBER).

Fix FA-2.3 (dictamen caracterizacion C, 2026-09-19):
  - FK corregida: OTHERMANAGER2.SEQUENCENUMBER.
  - 6 estados de token.
  - Multi-edge sin division economica.

Fuera de FA-2.3: canonicalizacion (FA-2.4), atribucion definitiva
(FA-2.3.bis), NIPC, breadth, clasificacion.
"""

from .cusip_resolver import (
    DEFAULT_EXCEPTIONS_PATH,
    FORBIDDEN_SOURCES,
    REQUIRED_COLUMNS,
    VALID_VERIFIED_BY,
    load_exceptions,
    resolve_batch,
    resolve_cusip,
)
from .relationships import (
    ALL_STATUSES,
    CANONICAL_KEY_COLUMNS,
    EDGE_COLUMNS,
    FORBIDDEN_TERMS,
    PROVISIONAL_FLAG,
    SEQ_MAX_DOMAIN,
    STATUS_INVALID_NONNUMERIC,
    STATUS_INVALID_OUT_OF_DOMAIN,
    STATUS_INVALID_ZERO,
    STATUS_NO_REFERENCE,
    STATUS_RESOLVED,
    STATUS_UNMAPPED_MISSING_OM2,
    build_canonical_relationship,
    build_filing_manager_index,
    build_om2_seq_index,
    build_provenance_index,
    check_edge_uniqueness,
    classify_token,
    compute_edge_metrics,
    compute_source_line_count,
    explode_othermanager_edges,
)
from .temporal_filter import CANONICAL_PERIOD_FIELD, FULL_PERIOD, filter_by_period

__all__ = [
    "filter_by_period", "CANONICAL_PERIOD_FIELD", "FULL_PERIOD",
    "load_exceptions", "resolve_cusip", "resolve_batch",
    "REQUIRED_COLUMNS", "FORBIDDEN_SOURCES", "VALID_VERIFIED_BY",
    "DEFAULT_EXCEPTIONS_PATH",
    "classify_token", "explode_othermanager_edges",
    "build_canonical_relationship",
    "build_filing_manager_index", "build_om2_seq_index",
    "build_provenance_index",
    "compute_edge_metrics", "check_edge_uniqueness",
    "compute_source_line_count",
    "ALL_STATUSES", "CANONICAL_KEY_COLUMNS", "EDGE_COLUMNS",
    "FORBIDDEN_TERMS", "PROVISIONAL_FLAG", "SEQ_MAX_DOMAIN",
    "STATUS_RESOLVED", "STATUS_NO_REFERENCE",
    "STATUS_INVALID_ZERO", "STATUS_INVALID_NONNUMERIC",
    "STATUS_INVALID_OUT_OF_DOMAIN", "STATUS_UNMAPPED_MISSING_OM2",
]
