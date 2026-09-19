"""Modulo identity - capa de identidad del framework 3 niveles (FA-2).

FA-2.1: temporal_filter.py (filtro canonico por PERIODOFREPORT).
FA-2.2: cusip_resolver.py (CUSIP -> ticker con vigencia temporal).
FA-2.3: relationships.py (canonical_reporting_relationship_key PROVISIONAL).

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
    CANONICAL_KEY_COLUMNS,
    FORBIDDEN_TERMS,
    PROVISIONAL_FLAG,
    assign_canonical_key,
    build_filing_manager_index,
    build_included_manager_index,
    build_provenance_index,
)
from .temporal_filter import CANONICAL_PERIOD_FIELD, FULL_PERIOD, filter_by_period

__all__ = [
    "filter_by_period", "CANONICAL_PERIOD_FIELD", "FULL_PERIOD",
    "load_exceptions", "resolve_cusip", "resolve_batch",
    "REQUIRED_COLUMNS", "FORBIDDEN_SOURCES", "VALID_VERIFIED_BY",
    "DEFAULT_EXCEPTIONS_PATH",
    "build_filing_manager_index", "build_included_manager_index",
    "build_provenance_index", "assign_canonical_key",
    "CANONICAL_KEY_COLUMNS", "FORBIDDEN_TERMS", "PROVISIONAL_FLAG",
]
