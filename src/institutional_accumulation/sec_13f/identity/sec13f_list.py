"""Parser SEC Official List of Section 13(f) Securities.

Dictamen habilitante: docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_
GATE0_SEC13F_DICTAMEN.md + INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_SEC13F_
MINIPROBE_DICTAMEN.md (PASS/CLOSED).

Layout fixed-width 80 chars, LF, segun dictamen mini-probe:
  Pos 01-09: CUSIP
  Pos 10:    Option Indicator ('*' o espacio)
  Pos 11-40: Issuer Name
  Pos 41-67: Issuer Description
  Pos 68-70: STATUS (3 chars: '*A*' add, '*D*' delete, blank no change)
  Pos 71-79: Blank
  Pos 80:    Misc / Unused. NO INTERPRETAR.

Responsabilidad limitada (dictamen Q-ESP-9):
  - Parsear fixed-width 80.
  - Extraer: CUSIP, option_indicator, issuer_name, issuer_description,
    status (3 chars).
  - Devolver section13f_eligible (5 estados) + status + option_indicator
    + raw_line.

NO resuelve ticker. NO resuelve security identity. Esa es responsabilidad
de cusip_resolver.py + security_identity.py + OpenFIGI fallback.

Estados internos (5, no booleano puro; dictamen v1.1 seccion 14):
  NOT_IN_LIST   CUSIP no aparece en la lista del periodo.
  DELETED       aparece con STATUS = *D*.
  ADDED         aparece con STATUS = *A*.
  ACTIVE        aparece con STATUS = blank.
  CONFLICT      mismo CUSIP con estados incompatibles.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LINE_WIDTH = 80
POS_CUSIP = (0, 9)
POS_OPTION = 9
POS_ISSUER_NAME = (10, 40)
POS_ISSUER_DESC = (40, 67)
POS_STATUS = (67, 70)
POS_BLANK = (70, 79)
POS_MISC = 79

STATUS_ADDED = "ADDED"
STATUS_DELETED = "DELETED"
STATUS_ACTIVE = "ACTIVE"
STATUS_NOT_IN_LIST = "NOT_IN_LIST"
STATUS_CONFLICT = "CONFLICT"

ALL_STATUSES = (
    STATUS_NOT_IN_LIST,
    STATUS_DELETED,
    STATUS_ADDED,
    STATUS_ACTIVE,
    STATUS_CONFLICT,
)

ELIGIBLE_STATES = (STATUS_ADDED, STATUS_ACTIVE)

COLUMNS = (
    "cusip",
    "option_indicator",
    "issuer_name",
    "issuer_description",
    "status_raw",
    "status",
    "raw_line",
)


def _classify_status(raw_status):
    """Clasifica STATUS de 3 chars (pos 68-70) a estado interno.

    *A*    -> ADDED
    *D*    -> DELETED
    blanco -> ACTIVE
    otro   -> None (anomalia de fuente)
    """
    s = (raw_status or "").strip()
    if s == "*A*":
        return STATUS_ADDED
    if s == "*D*":
        return STATUS_DELETED
    if s == "":
        return STATUS_ACTIVE
    return None


def parse_line(line):
    """Parsea una linea fixed-width 80. Devuelve dict o None.

    Ignora la posicion 80 (Misc/Unused, prohibido interpretar).
    Lineas vacias (solo whitespace) devuelven None.
    """
    if line is None:
        return None
    line = line.rstrip("\n\r")
    if not line.strip():
        return None
    if len(line) < LINE_WIDTH:
        line = line.ljust(LINE_WIDTH)

    raw_status = line[POS_STATUS[0]:POS_STATUS[1]]
    status = _classify_status(raw_status)

    return {
        "cusip": line[POS_CUSIP[0]:POS_CUSIP[1]].strip(),
        "option_indicator": line[POS_OPTION],
        "issuer_name": line[POS_ISSUER_NAME[0]:POS_ISSUER_NAME[1]].strip(),
        "issuer_description": line[POS_ISSUER_DESC[0]:POS_ISSUER_DESC[1]].strip(),
        "status_raw": raw_status,
        "status": status,
        "raw_line": line,
    }


def parse_official_list_text(text):
    """Parsea el contenido completo del TXT oficial.

    Devuelve DataFrame con columnas COLUMNS + _line_no + _order.
    Filas malformadas: incluidas con status=None (anomalia preservada).
    Orden de aparicion preservado.
    """
    rows = []
    for i, line in enumerate(text.split("\n")):
        parsed = parse_line(line)
        if parsed is None:
            continue
        parsed["_line_no"] = i + 1
        rows.append(parsed)
    df = pd.DataFrame(rows, columns=list(COLUMNS) + ["_line_no"])
    df["_order"] = range(len(df))
    return df


def load_official_list(path):
    """Carga un fichero TXT oficial SEC 13(f).

    Devuelve DataFrame con columnas COLUMNS + _line_no + _order.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Official List no existe: " + str(path))
    text = path.read_text(encoding="utf-8")
    return parse_official_list_text(text)


def _resolve_one_status(sub):
    """Resuelve el estado de un CUSIP a partir de sus filas.

    Regla (dictamen mini-probe Q-MINI-1..3):
      - Todas blank (ACTIVE) -> ACTIVE.
      - *A* y no *D* -> ADDED.
      - *D* y no *A* -> DELETED.
      - *A* y *D* en el mismo periodo -> CONFLICT
        (no 'ultimo gana' silencioso).
      - Todas con status None (anomalia) -> CONFLICT.
    """
    statuses = sub["status"].tolist()
    non_null = [s for s in statuses if s is not None]
    if not non_null:
        return STATUS_CONFLICT
    has_a = STATUS_ADDED in non_null
    has_d = STATUS_DELETED in non_null
    has_act = STATUS_ACTIVE in non_null
    if has_a and has_d:
        return STATUS_CONFLICT
    if has_a:
        return STATUS_ADDED
    if has_d:
        return STATUS_DELETED
    if has_act:
        return STATUS_ACTIVE
    return STATUS_CONFLICT


def resolve_eligibility(cusips, official_df):
    """Resuelve section13f_eligible (5 estados) para una lista de CUSIPs.

    cusips: iterable de str (CUSIPs del 13F).
    official_df: DataFrame producido por load_official_list.

    Devuelve dict {cusip: {"status", "option_indicator", "eligible"}}.
    eligible = status in {ADDED, ACTIVE}.
    CUSIPs no presentes -> NOT_IN_LIST.
    Regla: 'ultimo registro gana' PROHIBIDA. *A* + *D* -> CONFLICT.
    """
    if official_df is None or official_df.empty:
        return {str(c).strip(): {
            "status": STATUS_NOT_IN_LIST,
            "option_indicator": None,
            "eligible": False,
        } for c in cusips}

    by_cusip = official_df.groupby("cusip", sort=False)
    resolved = {}
    for cusip in cusips:
        c = str(cusip).strip()
        if c in by_cusip.groups:
            sub = official_df.loc[by_cusip.groups[c]]
            st = _resolve_one_status(sub)
            opts = [o for o in sub["option_indicator"].tolist() if o and o.strip()]
            opt = opts[0] if opts else None
        else:
            st = STATUS_NOT_IN_LIST
            opt = None
        resolved[c] = {
            "status": st,
            "option_indicator": opt,
            "eligible": st in ELIGIBLE_STATES,
        }
    return resolved


def compute_eligibility_coverage(cusips, official_df):
    """Metricas agregadas de elegibilidad.

    Devuelve dict con:
      n_total, n_eligible, n_not_in_list, n_deleted, n_active,
      n_added, n_conflict, pct_eligible
    """
    from collections import Counter
    resolved = resolve_eligibility(cusips, official_df)
    c = Counter(v["status"] for v in resolved.values())
    n_total = len(resolved)
    n_eligible = c.get(STATUS_ADDED, 0) + c.get(STATUS_ACTIVE, 0)
    return {
        "n_total": n_total,
        "n_eligible": n_eligible,
        "n_not_in_list": c.get(STATUS_NOT_IN_LIST, 0),
        "n_deleted": c.get(STATUS_DELETED, 0),
        "n_active": c.get(STATUS_ACTIVE, 0),
        "n_added": c.get(STATUS_ADDED, 0),
        "n_conflict": c.get(STATUS_CONFLICT, 0),
        "pct_eligible": (n_eligible / n_total) if n_total > 0 else 0.0,
    }