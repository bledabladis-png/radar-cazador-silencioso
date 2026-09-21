"""B1 - Materializacion del TargetUniverse (A62BIS_B1_SUBFASE.md v4 §3.6).

Vincula el snapshot B2-PIT con el membership B1 por snapshot_row_uid.
NO por posicion fisica. NO por ticker.

TargetUniverse = representacion administrativa del TARGET contractual
para un periodo dado. Dominio P38 se construye a partir de aqui via
catalog_p38_adapter.

Contrato de pureza:
  NO lee ficheros.
  NO escribe ficheros.
  NO usa datetime.now().
  Deterministas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

from src.institutional_accumulation.identity import catalog_key as ck


@dataclass(frozen=True)
class TargetUniverse:
    """Universo administrativo del catalogo para un periodo.

    Campos:
      version_id           identificador del snapshot B2-PIT.
      period_end           fecha efectiva del universo (YYYY-MM-DD).
      catalog_version_id   identificador administrativo del catalogo.
      catalog_sha256       hash del catalogo fuente.
      declared_keys        set de catalog_key vigentes en este snapshot.
      ticker_by_key        catalog_key -> radar_ticker.
      figi_by_key          catalog_key -> share_class_figi.
      row_uid_by_key       catalog_key -> snapshot_row_uid.
      key_by_row_uid       snapshot_row_uid -> catalog_key.
    """
    version_id: str
    period_end: str
    catalog_version_id: str
    catalog_sha256: str
    declared_keys: Set[str]
    ticker_by_key: Dict[str, str]
    figi_by_key: Dict[str, str]
    row_uid_by_key: Dict[str, str]
    key_by_row_uid: Dict[str, str]

    def __post_init__(self):
        # invariantes estructurales
        n = len(self.declared_keys)
        assert len(self.ticker_by_key) == n
        assert len(self.figi_by_key) == n
        assert len(self.row_uid_by_key) == n
        assert len(self.key_by_row_uid) == n


class BuildTargetError(ValueError):
    """Error de construccion del TargetUniverse."""


def build_target(snapshot_df, membership_df, assignments_df, *,
                 version_id, period_end,
                 catalog_version_id, catalog_sha256):
    """Materializa el TargetUniverse (§3.6).

    Validaciones:
      - snapshot cumple B1_REQUIRED_COLUMNS.
      - catalog_key NOT NULL y UNICO por snapshot.
      - membership cubre TODOS los row_uid del snapshot.
      - catalog_key existe en assignments.

    Vinculacion por snapshot_row_uid, no por posicion.
    """
    ck.check_schema(snapshot_df)

    # 1. indice membership: (version_id, row_uid) -> catalog_key
    mem = membership_df[membership_df["version_id"].astype(str) == str(version_id)]
    if mem.empty:
        raise BuildTargetError(
            "membership vacio para version_id=" + repr(version_id)
        )
    mem = mem.drop_duplicates(subset=["snapshot_row_uid"], keep=False)
    mem_by_uid = {}
    for _, r in mem.iterrows():
        uid = str(r["snapshot_row_uid"]).strip()
        k = str(r["catalog_key"]).strip()
        mem_by_uid[uid] = k

    # 2. indice assignments: catalog_key -> exists
    assign_keys = set(
        assignments_df["catalog_key"].astype(str).str.strip().tolist()
    )

    # 3. recorrer snapshot
    cols = list(snapshot_df.columns)
    ticker_by_key = {}
    figi_by_key = {}
    row_uid_by_key = {}
    key_by_row_uid = {}
    declared_keys = set()

    for _, row in snapshot_df.iterrows():
        uid = ck.compute_snapshot_row_uid(row, cols)
        k = mem_by_uid.get(uid)
        if k is None:
            raise BuildTargetError(
                "membership no cubre row_uid=" + uid[:16] + "..."
            )
        if k in declared_keys:
            raise BuildTargetError(
                "catalog_key duplicado en snapshot: " + k
            )
        if k not in assign_keys:
            raise BuildTargetError(
                "catalog_key no existe en assignments: " + k
            )
        declared_keys.add(k)
        ticker_by_key[k] = str(row.get("radar_ticker") or "").strip()
        figi_by_key[k] = str(row.get("share_class_figi") or "").strip()
        row_uid_by_key[k] = uid
        key_by_row_uid[uid] = k

    return TargetUniverse(
        version_id=str(version_id),
        period_end=str(period_end),
        catalog_version_id=str(catalog_version_id),
        catalog_sha256=str(catalog_sha256),
        declared_keys=declared_keys,
        ticker_by_key=ticker_by_key,
        figi_by_key=figi_by_key,
        row_uid_by_key=row_uid_by_key,
        key_by_row_uid=key_by_row_uid,
    )


# --- A2 c2/5 (fix H-10.1, 2026-09-21): SSHPRNAMT efectivo por FIGI ---


def extract_sshprnamt_by_figi(infotable_df, figi_by_cusip):
    """Extrae SSHPRNAMT efectivo agregado por shareClassFIGI.

    Fuente: INFOTABLE del canonical_snapshot post-amendments (Q1)
    o del filing base (Q4). La agregacion es SUM por FIGI sobre
    todos los CUSIP que resuelven al mismo shareClassFIGI dentro
    del periodo (contrato P38 seccion 3.3: w(s) por security,
    no por manager).

    Parametros:
      infotable_df    DataFrame con columnas CUSIP y SSHPRNAMT.
      figi_by_cusip   dict {CUSIP: shareClassFIGI}. CUSIPs no
                      presentes se ignoran (no se infiere FIGI).

    Devuelve dict {shareClassFIGI: float}. Valores SSHPRNAMT no
    numericos o negativos se descartan. DataFrame vacio o sin
    columnas requeridas -> {}.
    """
    if infotable_df is None or len(infotable_df) == 0:
        return {}
    cols = list(infotable_df.columns)
    if "CUSIP" not in cols or "SSHPRNAMT" not in cols:
        return {}
    out = {}
    for _, row in infotable_df.iterrows():
        cusip = str(row.get("CUSIP") or "").strip()
        if not cusip:
            continue
        figi = figi_by_cusip.get(cusip)
        if not figi:
            continue
        raw_val = row.get("SSHPRNAMT")
        if raw_val is None:
            continue
        try:
            v = float(raw_val)
        except (TypeError, ValueError):
            continue
        if v < 0:
            continue
        out[figi] = out.get(figi, 0.0) + v
    return out
