"""Generador de CSV de B1 desde el snapshot B2-PIT.

Genera:
  data/mappings/catalog_assignments.csv  (1 fila por catalog_key)
  data/mappings/catalog_membership.csv   (1 fila por key en el snapshot)

Contrato (A62BIS_B1_SUBFASE.md v4 secciones 2-3):
  catalog_key := "radar_<YYYYMMDD_alta>_<NNNN>"    (0001-based, alfabetico)
  assigned_entity_id := "radar_entity_<NNNN>"
  snapshot_row_uid := sha256_hex(fila_canonica)
    fila_canonica serializada length-prefixed, columnas en orden alfabetico:
      "<len>:<nombre>|<len>:<valor>|" repetido.

Migracion inicial:
  valid_from = alta_date (fecha del catalogo, source_date del snapshot)
  valid_to = null
  predecessor_row_uid = null
  justification = "initial migration"

Deterministico. Idempotente. No modifica el snapshot.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SNAPSHOT_DIR = ROOT / "data" / "mappings" / "catalog_snapshots"
MANIFEST_PATH = ROOT / "data" / "mappings" / "catalog_manifest.json"
ASSIGNMENTS_PATH = ROOT / "data" / "mappings" / "catalog_assignments.csv"
MEMBERSHIP_PATH = ROOT / "data" / "mappings" / "catalog_membership.csv"

CATALOG_KEY_PREFIX = "radar"
ENTITY_ID_PREFIX = "radar_entity"

ASSIGNMENT_COLUMNS = (
    "catalog_key", "assigned_entity_id",
    "valid_from", "valid_to", "source", "reason",
)
MEMBERSHIP_COLUMNS = (
    "version_id", "catalog_key", "snapshot_row_uid",
    "predecessor_row_uid", "justification",
)


def _canonical_value(v):
    """Normaliza el valor de una celda para la serializacion canonica."""
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def canonical_serialization(row, columns):
    """Serializa una fila como "len:name|len:value|..." por columna
    alfabetica. len en bytes UTF-8."""
    parts = []
    for name in sorted(columns):
        val = _canonical_value(row.get(name))
        name_b = name.encode("utf-8")
        val_b = val.encode("utf-8")
        parts.append(str(len(name_b)) + ":" + name)
        parts.append(str(len(val_b)) + ":" + val)
        parts.append("|")
    return "".join(parts)


def compute_snapshot_row_uid(row, columns):
    """sha256 hex (64 chars) del canonical_serialization."""
    ser = canonical_serialization(row, columns)
    return hashlib.sha256(ser.encode("utf-8")).hexdigest()


def load_snapshot(snapshot_path):
    return pd.read_csv(snapshot_path, dtype=str)


def load_manifest(manifest_path=MANIFEST_PATH):
    import json
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def _numeric_suffix(i):
    """0001-based, 4 digitos."""
    return str(i + 1).zfill(4)


def build_assignments(df, alta_date):
    """242 filas, orden alfabetico por radar_ticker."""
    ordered = df.sort_values("radar_ticker").reset_index(drop=True)
    rows = []
    for i, _ in ordered.iterrows():
        suffix = _numeric_suffix(i)
        rows.append({
            "catalog_key": CATALOG_KEY_PREFIX + "_" + alta_date + "_" + suffix,
            "assigned_entity_id": ENTITY_ID_PREFIX + "_" + suffix,
            "valid_from": alta_date,
            "valid_to": "",
            "source": "radar_target_catalog",
            "reason": "initial migration from snapshot",
        })
    return pd.DataFrame(rows, columns=list(ASSIGNMENT_COLUMNS))


def build_membership(df, version_id, assignments_df, *,
                     prev_uid_by_key=None, prev_just_by_key=None):
    """242 filas, 1 por key.

    prev_uid_by_key: dict {catalog_key: uid_en_snapshot_anterior}. Si el
    UID calculado en el snapshot vigente difiere, predecessor_row_uid
    se rellena con el UID anterior.

    prev_just_by_key: dict {catalog_key: justification_previa_no_inicial}.
    Se hereda para preservar historia.
    """
    prev_uid_by_key = prev_uid_by_key or {}
    prev_just_by_key = prev_just_by_key or {}
    ordered = df.sort_values("radar_ticker").reset_index(drop=True)
    assign_ordered = assignments_df.reset_index(drop=True)
    cols = list(df.columns)

    rows = []
    for i, (_, row) in enumerate(ordered.iterrows()):
        key = assign_ordered.iloc[i]["catalog_key"]
        new_uid = compute_snapshot_row_uid(row, cols)
        old_uid = prev_uid_by_key.get(key)

        if old_uid is None or old_uid == new_uid:
            pred = ""
            just = prev_just_by_key.get(key, "initial migration")
        else:
            pred = old_uid
            just = prev_just_by_key.get(key, "snapshot UID change")

        rows.append({
            "version_id": version_id,
            "catalog_key": key,
            "snapshot_row_uid": new_uid,
            "predecessor_row_uid": pred,
            "justification": just,
        })
    return pd.DataFrame(rows, columns=list(MEMBERSHIP_COLUMNS))



def _previous_snapshot_version_id(current_vid):
    """Devuelve el version_id del snapshot inmediatamente anterior.

    Ordena el manifest por valid_from asc, localiza current_vid,
    devuelve el anterior. Si no hay anterior, devuelve None.
    """
    manifest = load_manifest()
    snaps = sorted(manifest.get("snapshots", []),
                   key=lambda s: s["valid_from"])
    ids = [s["version_id"] for s in snaps]
    if current_vid not in ids:
        return None
    i = ids.index(current_vid)
    if i == 0:
        return None
    return ids[i - 1]


def _pick_snapshot():
    """Selecciona el snapshot vigente mas reciente del manifest.

    P62 permite multiples snapshots. Los publicados no se borran:
    el vigente es el de valid_to=null con valid_from mas reciente.
    """
    manifest = load_manifest()
    snaps = manifest.get("snapshots", [])
    if not snaps:
        raise RuntimeError("manifest sin snapshots")
    live = [s for s in snaps if s.get("valid_to") is None]
    if not live:
        raise RuntimeError("manifest sin snapshot vigente (valid_to=null)")
    live.sort(key=lambda s: s["valid_from"], reverse=True)
    s = live[0]
    version_id = s["version_id"]
    csv_path = ROOT / "data" / "mappings" / s["csv_path"]
    alta_date = str(s["valid_from"]).replace("-", "")
    return csv_path, version_id, alta_date


def main():
    csv_path, version_id, alta_date = _pick_snapshot()
    df = load_snapshot(csv_path)

    # Assignments: inmutables (catalog_key no se reasigna, contrato A1).
    if not ASSIGNMENTS_PATH.exists():
        assignments = build_assignments(df, alta_date)
        assignments.to_csv(ASSIGNMENTS_PATH, index=False, lineterminator="\n")
        print("OK assignments (creados): " + str(len(assignments)))
    else:
        assignments = pd.read_csv(ASSIGNMENTS_PATH, dtype=str, keep_default_na=False)
        print("OK assignments (existentes, inmutables): " + str(len(assignments)))

    # Comparar contra el snapshot inmediatamente anterior del manifest.
    prev_uid_by_key = {}
    prev_just_by_key = {}
    prev_vid = _previous_snapshot_version_id(version_id)
    if prev_vid is not None:
        manifest = load_manifest()
        entry = next(s for s in manifest["snapshots"] if s["version_id"] == prev_vid)
        prev_csv = ROOT / "data" / "mappings" / entry["csv_path"]
        prev_df = load_snapshot(prev_csv)
        prev_ordered = prev_df.sort_values("radar_ticker").reset_index(drop=True)
        prev_cols = list(prev_df.columns)
        assign_ordered = assignments.reset_index(drop=True)
        for i, (_, prow) in enumerate(prev_ordered.iterrows()):
            key = assign_ordered.iloc[i]["catalog_key"]
            prev_uid_by_key[key] = compute_snapshot_row_uid(prow, prev_cols)
        print("    comparando contra snapshot anterior: " + prev_vid)

    # Si existe membership previa, heredar justifications no-iniciales
    # (preserva historia de migraciones previas sobre las mismas keys)
    if MEMBERSHIP_PATH.exists():
        prev_mem = pd.read_csv(MEMBERSHIP_PATH, dtype=str, keep_default_na=False)
        for _, r in prev_mem.iterrows():
            j = r.get("justification", "")
            if j and j != "initial migration":
                prev_just_by_key[r["catalog_key"]] = j

    membership = build_membership(
        df, version_id, assignments,
        prev_uid_by_key=prev_uid_by_key,
        prev_just_by_key=prev_just_by_key,
    )
    membership.to_csv(MEMBERSHIP_PATH, index=False, lineterminator="\n")

    print("OK membership:  " + str(len(membership)) + " filas")
    print("    version_id=" + version_id + " alta_date=" + alta_date)


if __name__ == "__main__":
    main()