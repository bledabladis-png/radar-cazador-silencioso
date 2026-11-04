from __future__ import annotations

import math

VALID = "VALID"
DEGRADED = "DEGRADED"
INVALID = "INVALID"

COVERAGE_MIN = 0.90


def _is_finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def derive_mte_state(futures_status, coverage, confidence):
    if futures_status == "UNAVAILABLE" and coverage is None:
        return INVALID
    coverage_bad = (coverage is None) or (isinstance(coverage, (int, float)) and coverage < COVERAGE_MIN)
    if futures_status != "OK" or coverage_bad or not _is_finite(confidence):
        return DEGRADED
    return VALID
