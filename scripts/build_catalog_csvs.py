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

ROOT = Path(r"D:\Macro_Sectorial")
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


def build_membership(df, version_id, assignments_df):
    """242 filas, 1 por key. predecessor=null, justification='initial migration'."""
    ordered = df.sort_values("radar_ticker").reset_index(drop=True)
    assign_ordered = assignments_df.reset_index(drop=True)
    cols = list(df.columns)
    rows = []
    for i, (_, row) in enumerate(ordered.iterrows()):
        rows.append({
            "version_id": version_id,
            "catalog_key": assign_ordered.iloc[i]["catalog_key"],
            "snapshot_row_uid": compute_snapshot_row_uid(row, cols),
            "predecessor_row_uid": "",
            "justification": "initial migration",
        })
    return pd.DataFrame(rows, columns=list(MEMBERSHIP_COLUMNS))


def _pick_snapshot():
    """Selecciona el snapshot referenciado por el manifest."""
    manifest = load_manifest()
    snaps = manifest.get("snapshots", [])
    if len(snaps) != 1:
        raise RuntimeError(
            "manifest debe contener exactamente 1 snapshot, tiene " + str(len(snaps))
        )
    s = snaps[0]
    version_id = s["version_id"]
    csv_path = ROOT / "data" / "mappings" / s["csv_path"]
    alta_date = str(s["valid_from"]).replace("-", "")
    return csv_path, version_id, alta_date


def main():
    csv_path, version_id, alta_date = _pick_snapshot()
    df = load_snapshot(csv_path)
    assignments = build_assignments(df, alta_date)
    membership = build_membership(df, version_id, assignments)

    assignments.to_csv(ASSIGNMENTS_PATH, index=False, lineterminator="\n")
    membership.to_csv(MEMBERSHIP_PATH, index=False, lineterminator="\n")

    print("OK assignments: " + str(len(assignments)) + " filas -> " + str(ASSIGNMENTS_PATH))
    print("OK membership:  " + str(len(membership)) + " filas -> " + str(MEMBERSHIP_PATH))
    print("    version_id=" + version_id + " alta_date=" + alta_date)


if __name__ == "__main__":
    main()