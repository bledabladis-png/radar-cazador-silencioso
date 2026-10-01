"""Generador de CSV de B1 desde el snapshot B2-PIT (contrato v5).

Genera:
  data/mappings/catalog_assignments.csv  (append-only, 1 fila por catalog_key)
  data/mappings/catalog_membership.csv   (acumulativo, 1 fila por (version_id, catalog_key))

Contrato (A62BIS_B1_SUBFASE_V5.md):
  catalog_key := "radar_<YYYYMMDD_alta>_<NNNN>"
  assigned_entity_id := "radar_entity_<NNNN>"
  snapshot_row_uid := sha256_hex(fila_canonica)

Invariantes v5:
  - catalog_key inmutable.
  - En cada version V, radar_ticker <-> catalog_key es biyeccion del universo activo.
  - share_class_figi es el ancla de reconciliacion entre versiones.
  - Sin dependencia de posicion fisica.
  - Idempotente.
  - Membership acumulativo.

Migracion v1 -> v2: reconstruye radar_ticker y share_class_figi usando
membership (fuente autoritativa: version_id + snapshot_row_uid -> snapshot).
NO usa posicion fisica.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.institutional_accumulation.identity.catalog_key import (
    compute_snapshot_row_uid,
    ASSIGNMENT_COLUMNS as CK_ASSIGNMENT_COLUMNS,
    MEMBERSHIP_COLUMNS as CK_MEMBERSHIP_COLUMNS,
)

SNAPSHOT_DIR = ROOT / "data" / "mappings" / "catalog_snapshots"
MANIFEST_PATH = ROOT / "data" / "mappings" / "catalog_manifest.json"
ASSIGNMENTS_PATH = ROOT / "data" / "mappings" / "catalog_assignments.csv"
MEMBERSHIP_PATH = ROOT / "data" / "mappings" / "catalog_membership.csv"

CATALOG_KEY_PREFIX = "radar"
ENTITY_ID_PREFIX = "radar_entity"

ASSIGNMENT_COLUMNS = tuple(CK_ASSIGNMENT_COLUMNS)
MEMBERSHIP_COLUMNS = tuple(CK_MEMBERSHIP_COLUMNS)

KEY_RE = re.compile(r"^radar_(\d{8})_(\d{4})$")


def load_snapshot(snapshot_path):
    return pd.read_csv(snapshot_path, dtype=str, keep_default_na=False)


def load_manifest(manifest_path=MANIFEST_PATH):
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def _write_csv(df, path):
    text = df.to_csv(index=False, lineterminator="\n")
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def _pick_snapshot():
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


def _snapshot_path_for_version(version_id):
    manifest = load_manifest()
    for s in manifest["snapshots"]:
        if s["version_id"] == version_id:
            return ROOT / "data" / "mappings" / s["csv_path"]
    raise RuntimeError("version_id no en manifest: " + version_id)


def _previous_snapshot_version_id(current_vid):
    manifest = load_manifest()
    snaps = sorted(manifest.get("snapshots", []), key=lambda s: s["valid_from"])
    ids = [s["version_id"] for s in snaps]
    if current_vid not in ids:
        return None
    i = ids.index(current_vid)
    if i == 0:
        return None
    return ids[i - 1]


def _uid_to_ticker_figi_by_version(version_id):
    """Para un version_id dado, devuelve dict {uid: (ticker, figi)}.

    Resuelve desde el snapshot del manifest usando compute_snapshot_row_uid.
    """
    snap = load_snapshot(_snapshot_path_for_version(version_id))
    cols = list(snap.columns)
    out = {}
    for _, row in snap.iterrows():
        uid = compute_snapshot_row_uid(row, cols)
        out[uid] = (str(row["radar_ticker"]).strip(),
                    str(row["share_class_figi"]).strip())
    return out


# --- Migracion de assignments v1 (sin radar_ticker/share_class_figi) a v2 ---

def _migrate_assignments_v1_to_v2(asg_v1, mem_v1):
    """Reconstruye radar_ticker y share_class_figi usando membership.

    membership -> (version_id, catalog_key, snapshot_row_uid).
    Para cada fila: resolver (ticker, figi) del snapshot de version_id via uid.
    """
    cache = {}
    key_to_attr = {}
    for _, r in mem_v1.iterrows():
        vid = str(r["version_id"]).strip()
        key = str(r["catalog_key"]).strip()
        uid = str(r["snapshot_row_uid"]).strip()
        if vid not in cache:
            cache[vid] = _uid_to_ticker_figi_by_version(vid)
        tf = cache[vid].get(uid)
        if tf is None:
            continue
        key_to_attr[key] = tf

    rows = []
    for _, r in asg_v1.iterrows():
        key = str(r["catalog_key"]).strip()
        tf = key_to_attr.get(key, ("", ""))
        rows.append({
            "catalog_key": key,
            "assigned_entity_id": str(r["assigned_entity_id"]),
            "radar_ticker": tf[0],
            "share_class_figi": tf[1],
            "valid_from": str(r["valid_from"]),
            "valid_to": str(r["valid_to"]),
            "source": str(r["source"]),
            "reason": str(r["reason"]),
        })
    return pd.DataFrame(rows, columns=list(ASSIGNMENT_COLUMNS))


def _migrate_membership_v1_to_v2(mem_v1):
    """Añade radar_ticker (resuelto via snapshot) y reordena columnas."""
    cache = {}
    rows = []
    for _, r in mem_v1.iterrows():
        vid = str(r["version_id"]).strip()
        key = str(r["catalog_key"]).strip()
        uid = str(r["snapshot_row_uid"]).strip()
        if vid not in cache:
            cache[vid] = _uid_to_ticker_figi_by_version(vid)
        tf = cache[vid].get(uid, ("", ""))
        rows.append({
            "version_id": vid,
            "catalog_key": key,
            "radar_ticker": tf[0],
            "snapshot_row_uid": uid,
            "predecessor_row_uid": str(r["predecessor_row_uid"]),
            "justification": str(r["justification"]),
        })
    return pd.DataFrame(rows, columns=list(MEMBERSHIP_COLUMNS))


# --- Algoritmo de assignments (no posicional) ---

def _next_suffix_for_date(existing_keys, date_prefix):
    pat = re.compile(r"^radar_" + re.escape(date_prefix) + r"_(\d{4})$")
    max_n = 0
    for k in existing_keys:
        m = pat.match(str(k))
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _next_entity_n(existing_entity_ids):
    """Contador GLOBAL monotono para assigned_entity_id (no por fecha)."""
    pat = re.compile(r"^radar_entity_(\d{4})$")
    max_n = 0
    for e in existing_entity_ids:
        m = pat.match(str(e))
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def merge_assignments(asg_prev, snap, alta_date, new_date_prefix):
    """Reconciliacion por share_class_figi. Devuelve el DataFrame completo."""
    by_figi = {}
    for _, r in asg_prev.iterrows():
        f = str(r.get("share_class_figi", "")).strip()
        if f:
            by_figi[f] = r

    snap_sorted = snap.sort_values("radar_ticker").reset_index(drop=True)
    existing_keys = list(asg_prev["catalog_key"]) if len(asg_prev) > 0 else []
    existing_ent = list(asg_prev["assigned_entity_id"]) if len(asg_prev) > 0 else []
    next_n = _next_suffix_for_date(existing_keys, new_date_prefix)
    next_e = _next_entity_n(existing_ent)

    updates = {}
    new_rows = []
    for _, row in snap_sorted.iterrows():
        ticker = str(row["radar_ticker"]).strip()
        figi = str(row["share_class_figi"]).strip()
        if figi in by_figi:
            k = str(by_figi[figi]["catalog_key"]).strip()
            if str(by_figi[figi]["radar_ticker"]).strip() != ticker:
                updates[k] = ticker
        else:
            suffix = str(next_n).zfill(4)
            entity_suffix = str(next_e).zfill(4)
            new_rows.append({
                "catalog_key": CATALOG_KEY_PREFIX + "_" + new_date_prefix + "_" + suffix,
                "assigned_entity_id": ENTITY_ID_PREFIX + "_" + entity_suffix,
                "radar_ticker": ticker,
                "share_class_figi": figi,
                "valid_from": alta_date,
                "valid_to": "",
                "source": "radar_target_catalog",
                "reason": "snapshot expansion " + alta_date,
            })
            next_n += 1
            next_e += 1

    out = asg_prev.copy()
    for k, t in updates.items():
        out.loc[out["catalog_key"] == k, "radar_ticker"] = t
    if new_rows:
        out = pd.concat(
            [out, pd.DataFrame(new_rows, columns=list(ASSIGNMENT_COLUMNS))],
            ignore_index=True,
        )
    return out


# --- Backfill de versiones historicas faltantes ---

def backfill_missing_versions(mem_prev, asg):
    """Anade a membership las versiones del manifest que falten.

    Itera snapshots por valid_from ASC. Para cada version:
      - Si ya esta en mem_prev: solo actualiza prev_uid_by_key.
      - Si falta: genera filas con predecessor = UID de la version
        inmediatamente anterior procesada (o vacio si es la primera).

    prev_uid_by_key se pobla SOLO con versiones ya procesadas, nunca
    con versiones futuras.
    """
    manifest = load_manifest()
    snaps = sorted(manifest.get("snapshots", []), key=lambda s: s["valid_from"])
    existing_vids = set(mem_prev["version_id"].astype(str)) if len(mem_prev) > 0 else set()

    # figi -> key y ticker -> key para alinear con assignments
    figi_to_key = {}
    ticker_to_key = {}
    for _, r in asg.iterrows():
        f = str(r.get("share_class_figi", "")).strip()
        if f:
            figi_to_key[f] = str(r["catalog_key"]).strip()
        t = str(r.get("radar_ticker", "")).strip()
        if t:
            ticker_to_key[t] = str(r["catalog_key"]).strip()

    # UIDs por version y key, solo de mem_prev
    mem_by_version = {}
    if len(mem_prev) > 0:
        for _, r in mem_prev.iterrows():
            vid = str(r["version_id"]).strip()
            mem_by_version.setdefault(vid, []).append(r)

    prev_uid_by_key = {}
    added_rows = []
    added_vids = set()

    for s in snaps:
        vid = s["version_id"]

        if vid in existing_vids:
            # Actualizar prev_uid_by_key con los UIDs de esta version
            for r in mem_by_version.get(vid, []):
                k = str(r["catalog_key"]).strip()
                prev_uid_by_key[k] = str(r["snapshot_row_uid"]).strip()
            continue

        # Version faltante: generar filas
        snap = load_snapshot(ROOT / "data" / "mappings" / s["csv_path"])
        cols = list(snap.columns)
        new_uid_by_key = {}
        for _, row in snap.iterrows():
            ticker = str(row["radar_ticker"]).strip()
            figi = str(row["share_class_figi"]).strip()
            key = figi_to_key.get(figi) or ticker_to_key.get(ticker)
            if key is None:
                continue
            uid = compute_snapshot_row_uid(row, cols)
            prev_uid = prev_uid_by_key.get(key)
            if prev_uid is None:
                pred = ""
                just = "initial migration"
            elif prev_uid == uid:
                pred = ""
                just = "snapshot content unchanged"
            else:
                pred = prev_uid
                just = "snapshot content changed"
            added_rows.append({
                "version_id": vid,
                "catalog_key": key,
                "radar_ticker": ticker,
                "snapshot_row_uid": uid,
                "predecessor_row_uid": pred,
                "justification": just,
            })
            new_uid_by_key[key] = uid
        prev_uid_by_key.update(new_uid_by_key)
        added_vids.add(vid)

    if not added_rows:
        return mem_prev
    new_df = pd.DataFrame(added_rows, columns=list(MEMBERSHIP_COLUMNS))
    # Ordenar todo por version_id normalizado (sin guiones) para que el
    # CSV quede cronologico ascendente.
    out = pd.concat([new_df, mem_prev], ignore_index=True)
    out = out.assign(_vk=out["version_id"].str.replace("-", "", regex=False))
    out = out.sort_values(["_vk", "catalog_key"]).drop(columns=["_vk"]).reset_index(drop=True)
    return out

# --- Membership acumulativo ---

def build_membership_for_version(snap, asg, version_id, mem_prev):
    """Devuelve SOLO las filas nuevas de membership para version_id."""
    if len(mem_prev) > 0 and version_id in set(mem_prev["version_id"].astype(str)):
        return pd.DataFrame(columns=list(MEMBERSHIP_COLUMNS))

    figi_to_key = {}
    ticker_to_key = {}
    for _, r in asg.iterrows():
        f = str(r.get("share_class_figi", "")).strip()
        if f:
            figi_to_key[f] = str(r["catalog_key"]).strip()
        t = str(r.get("radar_ticker", "")).strip()
        if t:
            ticker_to_key[t] = str(r["catalog_key"]).strip()

    prev_uid_by_key = {}
    prev_vid = _previous_snapshot_version_id(version_id)
    if prev_vid and len(mem_prev) > 0:
        sub = mem_prev[mem_prev["version_id"].astype(str) == prev_vid]
        for _, r in sub.iterrows():
            prev_uid_by_key[str(r["catalog_key"]).strip()] = str(r["snapshot_row_uid"]).strip()

    cols = list(snap.columns)
    rows = []
    seen_keys = set()
    for _, row in snap.iterrows():
        ticker = str(row["radar_ticker"]).strip()
        figi = str(row["share_class_figi"]).strip()
        key = figi_to_key.get(figi)
        if key is None:
            key = ticker_to_key.get(ticker)
        if key is None:
            raise RuntimeError("ticker %s sin catalog_key en assignments" % ticker)
        if key in seen_keys:
            raise RuntimeError("catalog_key duplicada en snapshot: " + key)
        seen_keys.add(key)

        uid = compute_snapshot_row_uid(row, cols)
        prev_uid = prev_uid_by_key.get(key)

        if prev_uid is None:
            pred = ""
            just = "initial migration"
        elif prev_uid == uid:
            pred = ""
            just = "snapshot content unchanged"
        else:
            pred = prev_uid
            just = "snapshot content changed"

        rows.append({
            "version_id": version_id,
            "catalog_key": key,
            "radar_ticker": ticker,
            "snapshot_row_uid": uid,
            "predecessor_row_uid": pred,
            "justification": just,
        })
    return pd.DataFrame(rows, columns=list(MEMBERSHIP_COLUMNS))


# --- main ---

def main():
    csv_path, version_id, alta_date = _pick_snapshot()
    snap = load_snapshot(csv_path)
    print("snapshot vigente: %s (%d filas)" % (version_id, len(snap)))

    # Leer membership primero (necesario para migrar assignments).
    if MEMBERSHIP_PATH.exists():
        mem_prev = pd.read_csv(MEMBERSHIP_PATH, dtype=str, keep_default_na=False)
        mem_migrated = False
        if "radar_ticker" not in mem_prev.columns:
            print("[migracion] membership v1 -> v2")
            mem_prev = _migrate_membership_v1_to_v2(mem_prev)
            mem_migrated = True
        else:
            print("membership previo: %d filas (schema v2)" % len(mem_prev))
    else:
        mem_prev = pd.DataFrame(columns=list(MEMBERSHIP_COLUMNS))
        mem_migrated = False
        print("membership previo: vacio")

    # Assignments (necesario antes del backfill: resuelve figi->key).
    if ASSIGNMENTS_PATH.exists():
        asg_prev = pd.read_csv(ASSIGNMENTS_PATH, dtype=str, keep_default_na=False)
        if "share_class_figi" not in asg_prev.columns:
            print("[migracion] assignments v1 -> v2")
            asg_prev = _migrate_assignments_v1_to_v2(asg_prev, mem_prev)
        else:
            print("assignments previos: %d filas (schema v2)" % len(asg_prev))
    else:
        asg_prev = pd.DataFrame(columns=list(ASSIGNMENT_COLUMNS))
        print("assignments previos: vacio")

    asg_new = merge_assignments(asg_prev, snap, alta_date, alta_date)
    n_new = len(asg_new) - len(asg_prev)
    _write_csv(asg_new, ASSIGNMENTS_PATH)
    print("OK assignments: %d filas (+%d)" % (len(asg_new), n_new))

    # Backfill de versiones historicas faltantes (idempotente).
    mem_before_bf = len(mem_prev)
    mem_prev = backfill_missing_versions(mem_prev, asg_new)
    n_bf = len(mem_prev) - mem_before_bf
    if n_bf > 0:
        mem_migrated = True
        print("[backfill] membership: +%d filas de versiones historicas" % n_bf)

    # Membership: acumular la version vigente si no esta.
    mem_v = build_membership_for_version(snap, asg_new, version_id, mem_prev)
    if len(mem_v) == 0:
        if mem_migrated:
            _write_csv(mem_prev, MEMBERSHIP_PATH)
            print("OK membership: %d filas (migrado v1->v2, version_id=%s ya presente)"
                  % (len(mem_prev), version_id))
        else:
            print("OK membership: %d filas (version_id=%s ya presente, sin cambios)"
                  % (len(mem_prev), version_id))
    else:
        mem_new = pd.concat([mem_prev, mem_v], ignore_index=True)
        _write_csv(mem_new, MEMBERSHIP_PATH)
        print("OK membership: %d filas (+%d)" % (len(mem_new), len(mem_v)))


if __name__ == "__main__":
    main()