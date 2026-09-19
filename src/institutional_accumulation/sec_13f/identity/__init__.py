"""Modulo identity - capa de identidad del framework 3 niveles (FA-2).

FA-2.1: temporal_filter.py (filtro canonico por PERIODOFREPORT).
FA-2.2: cusip_resolver.py (CUSIP -> ticker con vigencia temporal).
FA-2.3: relationships.py (Column 7 -> OTHERMANAGER2.SEQUENCENUMBER).
FA-2.4: amendments.py (canonical snapshot composicional).

Fuera de FA-2.4: NIPC, breadth, clasificacion, cross-validation N-PORT
(post Gate FA-2).
"""

from .amendments import (
    ALL_ANOMALIES,
    ALL_STATUSES,
    ALL_STRATEGIES,
    ANOMALY_AMBIGUOUS_BASE,
    ANOMALY_AMBIGUOUS_ORDER,
    ANOMALY_HR_WITH_AMENDMENT_FLAGS,
    ANOMALY_NT_AUGMENTED,
    ANOMALY_REVIEW_REQUIRED,
    OP_ADD,
    OP_NO_HOLDINGS,
    OP_REPLACE,
    STATUS_CANONICAL,
    STATUS_CANONICAL_COMPOSITE,
    STATUS_NO_HOLDINGS,
    STATUS_REVIEW_REQUIRED,
    STATUS_SOURCE_ANOMALY,
    STRATEGY_HR_CHAIN_RESTATEMENT,
    STRATEGY_HR_COMPOSITE,
    STRATEGY_HR_PLUS_NEW_HOLDINGS,
    STRATEGY_HR_PLUS_RESTATEMENT,
    STRATEGY_NOTICE_AMENDED,
    STRATEGY_REVIEW_REQUIRED,
    STRATEGY_SINGLE_HR,
    STRATEGY_SINGLE_NOTICE,
    apply_amendments,
    classify_strategy,
    compute_status_counts,
    compute_strategy_counts,
    detect_base_filing,
    order_filings,
)
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
    "CANONICAL_KEY_COLUMNS", "EDGE_COLUMNS",
    "FORBIDDEN_TERMS", "PROVISIONAL_FLAG", "SEQ_MAX_DOMAIN",
    "STATUS_RESOLVED", "STATUS_NO_REFERENCE",
    "STATUS_INVALID_ZERO", "STATUS_INVALID_NONNUMERIC",
    "STATUS_INVALID_OUT_OF_DOMAIN", "STATUS_UNMAPPED_MISSING_OM2",
    "order_filings", "classify_strategy", "detect_base_filing",
    "apply_amendments", "compute_strategy_counts", "compute_status_counts",
    "ALL_STRATEGIES", "ALL_STATUSES", "ALL_ANOMALIES",
    "STRATEGY_SINGLE_HR", "STRATEGY_SINGLE_NOTICE",
    "STRATEGY_HR_PLUS_RESTATEMENT", "STRATEGY_HR_PLUS_NEW_HOLDINGS",
    "STRATEGY_HR_CHAIN_RESTATEMENT", "STRATEGY_HR_COMPOSITE",
    "STRATEGY_NOTICE_AMENDED", "STRATEGY_REVIEW_REQUIRED",
    "STATUS_CANONICAL", "STATUS_CANONICAL_COMPOSITE",
    "STATUS_NO_HOLDINGS", "STATUS_SOURCE_ANOMALY", "STATUS_REVIEW_REQUIRED",
    "ANOMALY_NT_AUGMENTED", "ANOMALY_HR_WITH_AMENDMENT_FLAGS",
    "ANOMALY_AMBIGUOUS_ORDER", "ANOMALY_AMBIGUOUS_BASE",
    "ANOMALY_REVIEW_REQUIRED",
    "OP_REPLACE", "OP_ADD", "OP_NO_HOLDINGS",
]
