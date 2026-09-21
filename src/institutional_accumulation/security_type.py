"""P38 - Clasificacion de security_type desde TITLEOFCLASS.

Contrato: NIPC spec v1.4 seccion 3.3.
Dictamen habilitante: #61 (2026-09-21, Nivel B AUTHORIZED).

Politica conservadora congelada (dictamen #61):
  STRONG EVIDENCE       -> RESOLVED
  AMBIGUOUS/INSUFFICIENT-> UNRESOLVED
  ACTUAL CONTRADICTION  -> CONFLICT
  UNRESOLVED|CONFLICT   -> NO operational_universe

Prohibiciones del dictamen:
  - NO fallback optimista a RESOLVED_EQUITY.
  - NO marcas de gestor/vehiculo (ISHARES, SPDR, VANGUARD, etc.)
    como autoridad de clasificacion.
  - NO BDC automatico por token.
  - NO substring ingenuo (tokenizacion + fronteras).
  - NO OpenFIGI masivo.

Precedencia de marcadores (dictamen #61 seccion 11):
  1. marcador NON_EQUITY fuerte
  2. marcador fondo excluido (mutual/closed-end/money market)
  3. marcador EQUITY fuerte (anchor)
  4. marcador ambiguo -> UNRESOLVED
  5. UNRESOLVED

Contrato de pureza (spec 11.3):
  NO leen ficheros.
  NO escriben ficheros.
  NO usan datetime.now().
  Deterministas.
"""
from __future__ import annotations

import re


# --- security_type (spec 3.3) ---
SECURITY_TYPE_EQUITY = "EQUITY"
SECURITY_TYPE_NON_EQUITY = "NON_EQUITY"
SECURITY_TYPE_ETF = "ETF"
SECURITY_TYPE_UNKNOWN = "UNKNOWN"

ALL_SECURITY_TYPES = (
    SECURITY_TYPE_EQUITY,
    SECURITY_TYPE_NON_EQUITY,
    SECURITY_TYPE_ETF,
    SECURITY_TYPE_UNKNOWN,
)


# --- security_type_status (spec 3.3) ---
STATUS_RESOLVED_EQUITY = "RESOLVED_EQUITY"
STATUS_RESOLVED_NON_EQUITY = "RESOLVED_NON_EQUITY"
STATUS_UNRESOLVED = "UNRESOLVED"
STATUS_CONFLICT = "CONFLICT"

ALL_SECURITY_TYPE_STATUSES = (
    STATUS_RESOLVED_EQUITY,
    STATUS_RESOLVED_NON_EQUITY,
    STATUS_UNRESOLVED,
    STATUS_CONFLICT,
)


# --- security_type_source (spec 3.3) ---
SOURCE_TITLEOFCLASS_13F = "TITLEOFCLASS_13F"
SOURCE_OPENFIGI_METADATA = "OPENFIGI_METADATA"
SOURCE_INTERNAL_CURATED = "INTERNAL_CURATED"
SOURCE_COMBINED = "COMBINED"

ALL_SECURITY_TYPE_SOURCES = (
    SOURCE_TITLEOFCLASS_13F,
    SOURCE_OPENFIGI_METADATA,
    SOURCE_INTERNAL_CURATED,
    SOURCE_COMBINED,
)


# --- Normalizacion / tokenizacion ---
# Fronteras de token: espacios, guiones, barras, comas, puntos.
_SPLIT_RE = re.compile(r"[^A-Z0-9]+")

EMPTY_MARKERS = frozenset({
    "",
    "NAN",
    "<NA>",
    "NONE",
    "N/A",
    "(BLANK)",
    "EMPTY TITLE",
})


def _normalize(toc):
    """Mayusculas, sin acentos, colapsa espacios."""
    if toc is None:
        return ""
    s = str(toc).strip().upper()
    return s


def _tokenize(norm):
    """Convierte a frozenset de tokens.

    'MUTUAL FUNDS-EQUITIES' -> {'MUTUAL', 'FUNDS', 'EQUITIES'}
    """
    if not norm:
        return frozenset()
    parts = [p for p in _SPLIT_RE.split(norm) if p]
    return frozenset(parts)


# --- Marcadores NON_EQUITY fuertes (tokens exactos) ---
# Fuente: dictamen #61 seccion 4.1.
NON_EQUITY_TOKENS = frozenset({
    "CONVERTIBLE",
    "CVT",
    "NOTE",
    "DEB",
    "DEBENTURE",
    "BOND",
    "PFD",
    "PREFERRED",
    "PREFERENCE",
    "WARRANT",
    "WARRANTS",
    "RIGHT",
    "RIGHTS",
    "UNIT",
    "UNITS",
    "ETF",
    "ETP",
})


# --- Frases NON_EQUITY (substring con fronteras de token) ---
NON_EQUITY_PHRASES = (
    "CONVERTIBLE BOND",
    "PREFERRED STOCK",
    "W EXP",
)


# --- Fondos / money market (NON_EQUITY por taxonomia contractual) ---
FUND_EXCLUDED_PHRASES = (
    "CLOSED END FUND",
    "CLOSED END",
    "MUTUAL FUNDS",
    "MUTUAL FUND",
    "MONEY MARKET",
)


# --- Anclas EQUITY aprobadas (dictamen #61 seccion 3.1) ---
# Tokens exactos. NO incluye CL A, SHS, STOCK, EQUITY, ADR, REIT, NEW.
EQUITY_ANCHOR_TOKENS = frozenset({
    "COM",
    "COMMON",
    "ORD",
    "ORDINARY",
    # Dictamen #62 seccion 4: marcadores autorizados como candidatos
    # de alta confianza. ADR / SPONSORED ADR / SPONSORED ADS NO
    # autorizados (permanecen UNRESOLVED).
    "SHS",
    "COMM",
    "CMN",
    "STK",
})


# --- Frases EQUITY aprobadas ---
EQUITY_PHRASES = (
    "COMMON STOCK",
    "COMMON SHARES",
    "CAPITAL STOCK",
    "CAP STK",
    "ORD SHS",
    "ORDINARY SHARES",
    # Dictamen #62 seccion 4: frases compuestas autorizadas.
    "COMM STK",
    "CL A",
    "CL B",
    "SHS CL A",
    "SPON ADS",
)


# --- Marcadores ETF/fondo excluidos del universo EQUITY ---
# Nota: nombres de gestores (ISHARES, SPDR, VANGUARD) NO se usan
# como autoridad (dictamen #61 seccion 4.3).
ETF_EXCLUDED_TOKENS = frozenset({
    "ETF",
    "ETP",
})


def _has_token(tokens, token_set):
    return bool(tokens & token_set)


def _has_phrase(norm, phrase):
    """Busca phrase con fronteras de token (no substring ingenuo).

    Usa tokens unidos por espacio para normalizar separadores
    (-, ., /, ,) antes de comparar. Ej: 'MUTUAL FUNDS-EQUITIES'
    -> 'MUTUAL FUNDS EQUITIES', matchea frase 'MUTUAL FUNDS'.
    """
    parts = _SPLIT_RE.split(norm)
    joined = " ".join(p for p in parts if p)
    padded = " " + joined + " "
    return (" " + phrase + " ") in padded


def _has_any_phrase(norm, phrases):
    return any(_has_phrase(norm, p) for p in phrases)


def classify_title_of_class(title_of_class):
    """Clasifica TITLEOFCLASS segun spec 3.3 + dictamen #61 Nivel B.

    Precedencia:
      1. NON_EQUITY fuerte (token o frase)
      2. Fondo excluido (mutual/closed-end/money market)
      3. EQUITY anchor (token o frase)
      4. UNRESOLVED

    Devuelve (security_type, security_type_status).

    NO detecta CONFLICT: para eso usar resolve_security_type() con
    evidencia externa (OpenFIGI). CONFLICT solo aparece ante
    contradiccion real.
    """
    norm = _normalize(title_of_class)
    if norm.upper() in EMPTY_MARKERS:
        return (SECURITY_TYPE_UNKNOWN, STATUS_UNRESOLVED)

    tokens = _tokenize(norm)

    # 1. NON_EQUITY fuerte
    if _has_token(tokens, NON_EQUITY_TOKENS):
        return (SECURITY_TYPE_NON_EQUITY, STATUS_RESOLVED_NON_EQUITY)
    if _has_any_phrase(norm, NON_EQUITY_PHRASES):
        return (SECURITY_TYPE_NON_EQUITY, STATUS_RESOLVED_NON_EQUITY)

    # 2. Fondos excluidos
    if _has_any_phrase(norm, FUND_EXCLUDED_PHRASES):
        return (SECURITY_TYPE_NON_EQUITY, STATUS_RESOLVED_NON_EQUITY)

    # 3. EQUITY anchor
    if _has_token(tokens, EQUITY_ANCHOR_TOKENS):
        return (SECURITY_TYPE_EQUITY, STATUS_RESOLVED_EQUITY)
    if _has_any_phrase(norm, EQUITY_PHRASES):
        return (SECURITY_TYPE_EQUITY, STATUS_RESOLVED_EQUITY)

    # 4. Fallback
    return (SECURITY_TYPE_UNKNOWN, STATUS_UNRESOLVED)


def resolve_security_type(title_of_class, external_evidence=None):
    """Combina TITLEOFCLASS con evidencia externa (OpenFIGI u otra).

    Dictamen #61 seccion 5:
      CONFLICT solo ante dos evidencias efectivas contradictorias.

    Parametros:
      title_of_class: str | None
      external_evidence: dict | None, con clave 'security_type' opcional.
                         Valores: 'EQUITY' | 'NON_EQUITY' | 'ETF' | 'UNKNOWN'.

    Devuelve (security_type, security_type_status).

    Reglas:
      - Sin evidencia externa -> resultado de classify_title_of_class.
      - Ambas coinciden -> status reforzado (mantiene el de la fuente).
      - Una UNRESOLVED + otra resuelta -> la resuelta gana (COMBINED).
      - Ambas resueltas y contradicen -> CONFLICT.
    """
    title_type, title_status = classify_title_of_class(title_of_class)

    if not external_evidence:
        return (title_type, title_status)

    ext = external_evidence.get("security_type")
    if ext is None or ext == SECURITY_TYPE_UNKNOWN:
        return (title_type, title_status)

    ext_norm = str(ext).strip().upper()
    if ext_norm not in ALL_SECURITY_TYPES:
        return (title_type, title_status)

    title_resolved = title_status in (
        STATUS_RESOLVED_EQUITY,
        STATUS_RESOLVED_NON_EQUITY,
    )

    if not title_resolved:
        # Title no resuelve: aceptar external.
        if ext_norm == SECURITY_TYPE_EQUITY:
            return (SECURITY_TYPE_EQUITY, STATUS_RESOLVED_EQUITY)
        if ext_norm == SECURITY_TYPE_NON_EQUITY:
            return (SECURITY_TYPE_NON_EQUITY, STATUS_RESOLVED_NON_EQUITY)
        return (title_type, title_status)

    # Title resuelve. Ver si external coincide.
    if ext_norm == title_type:
        return (title_type, title_status)

    # Contradiccion real.
    return (SECURITY_TYPE_UNKNOWN, STATUS_CONFLICT)


def coverage_stats(records):
    """Cuenta estados para reporte pct_unresolved (dictamen #61 seccion 8).

    records: iterable de tuplas (security_type, status) o dicts.

    Devuelve dict con conteos y porcentajes.
    """
    counts = {
        STATUS_RESOLVED_EQUITY: 0,
        STATUS_RESOLVED_NON_EQUITY: 0,
        STATUS_UNRESOLVED: 0,
        STATUS_CONFLICT: 0,
    }
    total = 0
    for rec in records:
        if isinstance(rec, dict):
            status = rec.get("security_type_status")
        else:
            status = rec[1] if len(rec) > 1 else None
        if status in counts:
            counts[status] += 1
        total += 1

    if total == 0:
        return {
            "total": 0,
            "counts": dict(counts),
            "pct_resolved": 0.0,
            "pct_unresolved": 0.0,
            "pct_conflict": 0.0,
            "pct_operational": 0.0,
        }

    resolved = counts[STATUS_RESOLVED_EQUITY] + counts[STATUS_RESOLVED_NON_EQUITY]
    operational = counts[STATUS_RESOLVED_EQUITY]  # solo EQUITY entra
    return {
        "total": total,
        "counts": dict(counts),
        "pct_resolved": round(100.0 * resolved / total, 4),
        "pct_unresolved": round(100.0 * counts[STATUS_UNRESOLVED] / total, 4),
        "pct_conflict": round(100.0 * counts[STATUS_CONFLICT] / total, 4),
        "pct_operational": round(100.0 * operational / total, 4),
    }