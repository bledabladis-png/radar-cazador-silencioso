"""Modulo identity - capa de identidad del framework 3 niveles (FA-2).

FA-2.1 alcance:
  - temporal_filter.py: filtro canonico por SUBMISSION.PERIODOFREPORT.

Fuera de FA-2.1:
  - CUSIP resolver (FA-2.2).
  - Reporting relationships (FA-2.3).
  - Amendments (FA-2.4).
  - NIPC, breadth, clasificacion (post Gate FA-2).
"""

from .temporal_filter import filter_by_period, CANONICAL_PERIOD_FIELD, FULL_PERIOD

__all__ = ["filter_by_period", "CANONICAL_PERIOD_FIELD", "FULL_PERIOD"]
