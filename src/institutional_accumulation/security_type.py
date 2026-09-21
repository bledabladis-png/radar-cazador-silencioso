"""P38 - Clasificacion de security_type desde TITLEOFCLASS.

Contrato: NIPC spec v1.4 seccion 3.3.
Dictamen habilitante: A.6.2-bis (pendiente; alcance inicial Nivel A).

Alcance Nivel A (inicial):
  Solo detecta NON_EQUITY por marcadores explicitamente documentados
  en la spec (seccion 3.3: caso explicito + enum NON_EQUITY):
    - CONVERTIBLE   caso explicito spec 3.3, cubre GHISALLO
    - NOTE          bonos (enum NON_EQUITY)
    - PFD           preferentes (enum NON_EQUITY)
    - PREFERRED     preferentes (variante ortografica)
  Todo lo demas queda UNKNOWN + UNRESOLVED.

Limitaciones conocidas (a revisar con auditor):
  - TITLEOFCLASS tiene 16.887 valores unicos en Q1 2026. Es texto
    libre del filer. Nivel A NO clasifica EQUITY (no hay marcadores
    positivos). El filtro seccion 5.4 (== RESOLVED_EQUITY) produce
    universo vacio con Nivel A. Requiere Nivel B (marcadores EQUITY)
    o Nivel C (OpenFIGI masivo) para ser funcional.
  - TITLEOFCLASS NO es autoridad unica (Q-NIPC-8). Los marcadores
    son input, no decision automatica.
  - Marcadores incompletos: warrants, rights, units, ETFs, funds,
    ADRs, depositary receipts no cubiertos en Nivel A.

Contrato de pureza (spec 11.3):
  NO leen ficheros.
  NO escriben ficheros.
  NO usan datetime.now().
  Deterministas.
"""
from __future__ import annotations


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


# --- Marcadores NON_EQUITY (Nivel A) ---
# Fuente: spec NIPC v1.4 seccion 3.3 (caso explicito + enum NON_EQUITY).

NON_EQUITY_SUBSTRING_MARKERS = (
    "CONVERTIBLE",
    "PFD",
    "PREFERRED",
)

NON_EQUITY_PREFIX_MARKERS = (
    "NOTE ",
)

NON_EQUITY_EXACT_MARKERS = (
    "NOTE",
)


EMPTY_MARKERS = frozenset({
    "",
    "NAN",
    "<NA>",
    "NONE",
    "N/A",
    "(BLANK)",
})


def classify_title_of_class(title_of_class):
    """Clasifica TITLEOFCLASS segun spec seccion 3.3 (Nivel A).

    Parametros:
      title_of_class: valor de la columna TITLEOFCLASS (str | None).

    Devuelve tupla (security_type, security_type_status).

    Nivel A: solo detecta NON_EQUITY por marcadores documentados.
    Todo lo demas queda (UNKNOWN, UNRESOLVED). No clasifica EQUITY.
    """
    if title_of_class is None:
        return (SECURITY_TYPE_UNKNOWN, STATUS_UNRESOLVED)

    s = str(title_of_class).strip()
    if s.upper() in EMPTY_MARKERS:
        return (SECURITY_TYPE_UNKNOWN, STATUS_UNRESOLVED)

    u = s.upper()

    for marker in NON_EQUITY_SUBSTRING_MARKERS:
        if marker in u:
            return (SECURITY_TYPE_NON_EQUITY, STATUS_RESOLVED_NON_EQUITY)

    for marker in NON_EQUITY_PREFIX_MARKERS:
        if u.startswith(marker):
            return (SECURITY_TYPE_NON_EQUITY, STATUS_RESOLVED_NON_EQUITY)

    if u in NON_EQUITY_EXACT_MARKERS:
        return (SECURITY_TYPE_NON_EQUITY, STATUS_RESOLVED_NON_EQUITY)

    return (SECURITY_TYPE_UNKNOWN, STATUS_UNRESOLVED)