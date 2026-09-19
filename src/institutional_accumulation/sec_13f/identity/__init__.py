"""Modulo identity - capa de identidad del framework 3 niveles (FA-2).

FA-2.1 alcance:
  - temporal_filter.py: filtro canonico por SUBMISSION.PERIODOFREPORT.

FA-2.2 alcance:
  - cusip_resolver.py: resolver CUSIP -> ticker con vigencia temporal.

Fuera de FA-2.2:
  - Reporting relationships (FA-2.3).
  - Amendments (FA-2.4).
  - NIPC, breadth, clasificacion (post Gate FA-2).
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
from .temporal_filter import CANONICAL_PERIOD_FIELD, FULL_PERIOD, filter_by_period

__all__ = [
    "filter_by_period",
    "CANONICAL_PERIOD_FIELD",
    "FULL_PERIOD",
    "load_exceptions",
    "resolve_cusip",
    "resolve_batch",
    "REQUIRED_COLUMNS",
    "FORBIDDEN_SOURCES",
    "VALID_VERIFIED_BY",
    "DEFAULT_EXCEPTIONS_PATH",
]
