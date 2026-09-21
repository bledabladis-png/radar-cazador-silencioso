"""B1 - Identidad administrativa del catalogo (A62BIS_B1_SUBFASE.md v4).

Cubre:
  seccion 2  A1 - catalog_key inmutable
  seccion 3  A1-membership v4 (B1-NEW-6 + B1-NEW-7)
  seccion 8  B1-NEW-9 contrato de entrada B1 <-> B2-PIT

Reglas normativas:
  catalog_key := "radar_<YYYYMMDD_alta>_<NNNN>"
  assigned_entity_id := "radar_entity_<NNNN>"
  K -> assigned_entity_id es 1:1 para toda la vida de K.
  K NO se reasigna jamas.

Contrato de pureza:
  NO lee ficheros salvo las funciones load_* explicitas.
  NO escribe ficheros.
  NO usa datetime.now().
  Deterministas.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd


# --- Formatos ---
CATALOG_KEY_RE = re.compile(r"^radar_\d{8}_\d{4}$")
ENTITY_ID_RE = re.compile(r"^radar_entity_\d{4}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


# --- Contrato de entrada (B1-NEW-9) ---
B1_REQUIRED_COLUMNS = frozenset({
    "radar_ticker",
    "share_class_figi",
})


# --- Columnas de los CSV B1 ---
ASSIGNMENT_COLUMNS = (
    "catalog_key", "assigned_entity_id",
    "valid_from", "valid_to", "source", "reason",
)
MEMBERSHIP_COLUMNS = (
    "version_id", "catalog_key", "snapshot_row_uid",
    "predecessor_row_uid", "justification",
)
ATTEMPTED_COLUMNS = (
    "catalog_key", "attempted_entity_id",
    "attempted_at", "reason", "source",
)


# --- Codigos de error ---
ERR_CATALOG_KEY_REASSIGNED = "CATALOG_KEY_REASSIGNED"
ERR_CATALOG_KEY_RETIRED_REACTIVATED = "CATALOG_KEY_RETIRED_REACTIVATED"
ERR_CATALOG_KEY_NOT_VALID_FOR_SNAPSHOT = "CATALOG_KEY_NOT_VALID_FOR_SNAPSHOT_INTERVAL"
ERR_MISSING_VERSION_IN_MANIFEST = "MISSING_VERSION_IN_MANIFEST"
ERR_MISSING_CATALOG_KEY_IN_ASSIGNMENTS = "MISSING_CATALOG_KEY_IN_ASSIGNMENTS"
ERR_DUPLICATE_CATALOG_KEY_IN_SNAPSHOT = "DUPLICATE_CATALOG_KEY_IN_SNAPSHOT"
ERR_ROW_UID_NOT_IN_SNAPSHOT = "ROW_UID_NOT_IN_SNAPSHOT"
ERR_PREDECESSOR_ROW_UID_BROKEN_CHAIN = "PREDECESSOR_ROW_UID_BROKEN_CHAIN"
ERR_B1_SCHEMA_ERROR = "B1_SCHEMA_ERROR"


# --- Helpers ---

def _to_date(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    if not s:
        return None
    try:
        return pd.Timestamp(s).normalize()
    except Exception:
        return None


def is_valid_catalog_key(s):
    return isinstance(s, str) and bool(CATALOG_KEY_RE.match(s))


def is_valid_entity_id(s):
    return isinstance(s, str) and bool(ENTITY_ID_RE.match(s))


def is_valid_sha256(s):
    return isinstance(s, str) and bool(SHA256_RE.match(s))


# --- Serializacion canonica (B1-NEW-7) ---

def _canonical_value(v):
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
    alfabetica. len en bytes UTF-8. Length-prefixed (evita ambiguedad)."""
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


# --- Contrato de entrada (B1-NEW-9) ---

def check_schema(df):
    """Verifica B1_REQUIRED_COLUMNS. Raise ValueError si falta alguna."""
    missing = sorted(B1_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            ERR_B1_SCHEMA_ERROR + ": columnas requeridas ausentes: "
            + ", ".join(missing)
        )
    return True


# --- Loaders ---

def load_assignments(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ASSIGNMENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "catalog_assignments.csv sin columnas: " + ", ".join(missing)
        )
    return df


def load_membership(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in MEMBERSHIP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "catalog_membership.csv sin columnas: " + ", ".join(missing)
        )
    return df


def load_attempted(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in ATTEMPTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "catalog_reassignments_attempted.csv sin columnas: "
            + ", ".join(missing)
        )
    return df


# --- Validacion A1 (seccion 2.6) ---

def validate_assignment(assignments_df, attempted_df=None):
    """Devuelve dict {catalog_key: error_code}.

    Errores:
      CATALOG_KEY_REASSIGNED
        K con assigned_entity_id distinto en assignments_df, o
        intento en attempted_df con attempted_entity_id != original.
      CATALOG_KEY_RETIRED_REACTIVATED
        K con valid_to caducado aparece en attempted_df con
        attempted_at posterior (reactivacion).
    """
    errors = {}
    if assignments_df is None or assignments_df.empty:
        return errors

    # 1. catalog_key unico en assignments (unica alta inicial).
    keys = assignments_df["catalog_key"].tolist()
    if len(set(keys)) != len(keys):
        for k in keys:
            if keys.count(k) > 1:
                errors[k] = ERR_CATALOG_KEY_REASSIGNED

    # 2. indice key -> (entity_id, valid_to)
    idx = {}
    for _, r in assignments_df.iterrows():
        k = str(r["catalog_key"]).strip()
        idx[k] = {
            "entity_id": str(r["assigned_entity_id"]).strip(),
            "valid_to": _to_date(r.get("valid_to")),
        }

    # 3. intentos
    if attempted_df is not None and not attempted_df.empty:
        for _, r in attempted_df.iterrows():
            k = str(r["catalog_key"]).strip()
            if k not in idx:
                continue
            att_ent = str(r["attempted_entity_id"]).strip()
            orig = idx[k]
            if att_ent and att_ent != orig["entity_id"]:
                errors[k] = ERR_CATALOG_KEY_REASSIGNED
                continue
            # retirada + reintento posterior
            att_at = _to_date(r.get("attempted_at"))
            if orig["valid_to"] is not None and att_at is not None:
                if att_at > orig["valid_to"]:
                    errors[k] = ERR_CATALOG_KEY_RETIRED_REACTIVATED

    return errors


# --- Validacion membership (seccion 3.5) ---

def _manifest_versions(manifest):
    snaps = manifest.get("snapshots", []) if isinstance(manifest, dict) else []
    out = {}
    for s in snaps:
        vid = s.get("version_id")
        if vid:
            out[vid] = s
    return out


def _interval_covers(snapshot_row, assignment_row):
    """Cobertura completa (B1-NEW-6):
       assignment.valid_from <= snapshot.valid_from
       AND (assignment.valid_to IS NULL
            OR (snapshot.valid_to IS NOT NULL
                AND snapshot.valid_to <= assignment.valid_to))
    """
    svf = _to_date(snapshot_row.get("valid_from"))
    svt = _to_date(snapshot_row.get("valid_to"))
    avf = _to_date(assignment_row.get("valid_from"))
    avt = _to_date(assignment_row.get("valid_to"))

    if avf is None or svf is None:
        return False
    if avf > svf:
        return False
    if avt is None:
        return True
    if svt is None:
        return False
    return svt <= avt


def validate_membership(membership_df, assignments_df, manifest,
                        *, snapshot_loader=None):
    """Devuelve dict {(version_id, catalog_key): error_code}.

    snapshot_loader: callable(version_id) -> DataFrame | None.
      Si se pasa, valida ROW_UID_NOT_IN_SNAPSHOT y cadenas predecessor.
    """
    errors = {}
    if membership_df is None or membership_df.empty:
        return errors

    manifest_versions = _manifest_versions(manifest)

    assign_idx = {}
    if assignments_df is not None and not assignments_df.empty:
        for _, r in assignments_df.iterrows():
            k = str(r["catalog_key"]).strip()
            assign_idx[k] = r

    # Cache de snapshots cargados y de row_uids por version
    _snap_cache = {}
    _row_uids_cache = {}

    def _load_snap(vid):
        if vid in _snap_cache:
            return _snap_cache[vid]
        df = None
        if snapshot_loader is not None:
            df = snapshot_loader(vid)
        _snap_cache[vid] = df
        return df

    def _row_uids(vid):
        if vid in _row_uids_cache:
            return _row_uids_cache[vid]
        df = _load_snap(vid)
        if df is None:
            _row_uids_cache[vid] = set()
            return _row_uids_cache[vid]
        cols = list(df.columns)
        uids = set()
        for _, row in df.iterrows():
            uids.add(compute_snapshot_row_uid(row, cols))
        _row_uids_cache[vid] = uids
        return _row_uids_cache[vid]

    # Indice por version: keys y sus row_uids
    keys_by_version = {}
    for _, r in membership_df.iterrows():
        vid = str(r["version_id"]).strip()
        k = str(r["catalog_key"]).strip()
        keys_by_version.setdefault(vid, []).append(k)

    # Predecesores: (catalog_key -> {version_id -> row_uid})
    pred_by_key = {}
    for _, r in membership_df.iterrows():
        vid = str(r["version_id"]).strip()
        k = str(r["catalog_key"]).strip()
        ru = str(r["snapshot_row_uid"]).strip()
        pred_by_key.setdefault(k, {})[vid] = ru

    # --- Chequeos por fila ---
    for _, r in membership_df.iterrows():
        vid = str(r["version_id"]).strip()
        k = str(r["catalog_key"]).strip()
        ru = str(r["snapshot_row_uid"]).strip()
        pred = str(r.get("predecessor_row_uid") or "").strip()
        err_key = (vid, k)

        # version en manifest
        if vid not in manifest_versions:
            errors[err_key] = ERR_MISSING_VERSION_IN_MANIFEST
            continue

        # catalog_key en assignments
        if k not in assign_idx:
            errors[err_key] = ERR_MISSING_CATALOG_KEY_IN_ASSIGNMENTS
            continue

        # duplicado catalog_key en snapshot
        if keys_by_version.get(vid, []).count(k) > 1:
            errors[err_key] = ERR_DUPLICATE_CATALOG_KEY_IN_SNAPSHOT
            continue

        # cobertura temporal
        if not _interval_covers(manifest_versions[vid], assign_idx[k]):
            errors[err_key] = ERR_CATALOG_KEY_NOT_VALID_FOR_SNAPSHOT
            continue

        # row_uid en snapshot (si hay loader)
        if snapshot_loader is not None:
            if ru not in _row_uids(vid):
                errors[err_key] = ERR_ROW_UID_NOT_IN_SNAPSHOT
                continue

        # predecessor chain
        if pred:
            prior_versions = [
                v for v in pred_by_key.get(k, {}).keys() if v < vid
            ]
            if not prior_versions:
                errors[err_key] = ERR_PREDECESSOR_ROW_UID_BROKEN_CHAIN
                continue
            pred_uids = {
                pred_by_key[k][v] for v in prior_versions
            }
            if pred not in pred_uids:
                errors[err_key] = ERR_PREDECESSOR_ROW_UID_BROKEN_CHAIN

    return errors