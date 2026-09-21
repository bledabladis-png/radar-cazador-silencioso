"""P63/B3 - Derivacion de los 3 timestamps de una observacion 13F.

Contrato semantico: docs/auditoria/iae/NIPC_CONTRATOS_SEMANTICOS_v1.md seccion 12.
Dictamen habilitante: A.6.2-bis-B3 (#48/#49).

Los 3 timestamps son:
  period_end      fecha de cierre del periodo 13F observado
                  (SUBMISSION.PERIODOFREPORT).
  filing_date     fecha de presentacion del filing que contiene la
                  observacion (SUBMISSION.FILING_DATE).
  knowledge_date  fecha en que la observacion era publicamente conocida.

Regla opcion A (DICTAMENES #45 seccion 11):
  knowledge_date == filing_date.
  Ningun registro aparece como conocido antes del filing que lo hace
  publico. Amendments incluidos.

Este modulo opera sobre el resultado de apply_amendments() (dict con
canonical_snapshot + applied_accessions + lineage + ...).
"""
from __future__ import annotations

import pandas as pd


def _to_date_str(v):
    """Normaliza un timestamp/date a 'YYYY-MM-DD' o None si vacio."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        return None
    return str(pd.Timestamp(v).date())


def derive_timestamps(snapshot, accession):
    """Dado un snapshot y un accession, devuelve los 3 timestamps.

    Opcion A: knowledge_date == filing_date.

    Parametros:
      snapshot   dict retornado por apply_amendments().
      accession  str con el ACCESSION_NUMBER a resolver.

    Devuelve dict con:
      period_end      'YYYY-MM-DD' | None
      filing_date     'YYYY-MM-DD' | None
      knowledge_date  'YYYY-MM-DD' | None

    Si el accession no esta en el snapshot, los 3 valores son None.
    """
    none_result = {"period_end": None, "filing_date": None, "knowledge_date": None}
    sub = snapshot.get("SUBMISSION") if isinstance(snapshot, dict) else None
    if sub is None or sub.empty:
        return dict(none_result)
    match = sub[sub["ACCESSION_NUMBER"].astype(str) == str(accession)]
    if match.empty:
        return dict(none_result)
    row = match.iloc[0]
    filing_date = _to_date_str(row.get("FILING_DATE"))
    return {
        "period_end": _to_date_str(row.get("PERIODOFREPORT")),
        "filing_date": filing_date,
        "knowledge_date": filing_date,
    }


def enrich_positions_with_timestamps(
    positions,
    snapshot,
    accession_col="ACCESSION_NUMBER",
):
    """Enriquece un DataFrame de posiciones con los 3 timestamps.

    Anade columnas:
      period_end      fecha cierre periodo
      filing_date     fecha presentacion
      knowledge_date  == filing_date (opcion A)

    Filas cuyo accession no esta en snapshot quedan con None en los 3.
    NO muta el DataFrame de entrada (copia defensiva).

    Parametros:
      positions      DataFrame con accession_col.
      snapshot       dict de apply_amendments().
      accession_col  nombre de columna con accession (default ACCESSION_NUMBER).
    """
    out = positions.copy()
    if accession_col not in out.columns:
        raise ValueError(
            "positions no contiene la columna " + accession_col
        )

    sub = snapshot.get("SUBMISSION") if isinstance(snapshot, dict) else None
    if sub is None or sub.empty:
        out["period_end"] = None
        out["filing_date"] = None
        out["knowledge_date"] = None
        return out

    sub_clean = sub[["ACCESSION_NUMBER", "PERIODOFREPORT", "FILING_DATE"]].copy()
    sub_clean["ACCESSION_NUMBER"] = sub_clean["ACCESSION_NUMBER"].astype(str)
    sub_clean = sub_clean.drop_duplicates("ACCESSION_NUMBER").set_index("ACCESSION_NUMBER")

    accs = out[accession_col].astype(str)
    out["period_end"] = accs.map(sub_clean["PERIODOFREPORT"]).map(_to_date_str)
    out["filing_date"] = accs.map(sub_clean["FILING_DATE"]).map(_to_date_str)
    out["knowledge_date"] = out["filing_date"]
    return out