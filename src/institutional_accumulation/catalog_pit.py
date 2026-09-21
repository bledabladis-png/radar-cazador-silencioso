"""B2-PIT: infraestructura temporal del catalogo.

Frontera arquitectonica (dictamen #52, seccion 5):
    B2-PIT responde: "¿que catalogo era valido en una fecha?"
    B2-PIT NO responde: "¿que TARGET contractual representa ese catalogo?"
    B2-PIT NO define identidad administrativa (catalog_key). Eso es B1.

Alcance:
    snapshot materializado + manifest + sha256 externo
    valid_from / valid_to (semiabiertos)
    target_catalog_as_of(period_end)
    integridad + corrupcion + ambiguedad
    no backdating

Contrato de pureza (spec 11.3):
    NO leen ficheros en fase de calculo salvo los propios snapshots.
    NO usan datetime.now() como fecha de observacion.
    NO escriben ficheros.
    Deterministas.

Los snapshots se publican en una fase administrativa externa (no en
runtime). Este modulo solo lee.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


MANIFEST_SCHEMA_VERSION = 1
SNAPSHOT_FILENAME_TEMPLATE = "snapshot_{version_id}.csv"
SNAPSHOT_SHA_FILENAME_TEMPLATE = "snapshot_{version_id}.sha256"


# --- Excepciones ---

class CatalogPitError(Exception):
    """Base de errores de B2-PIT."""


class CatalogNotAvailable(CatalogPitError):
    """No existe snapshot que cubra period_end."""


class CatalogAmbiguous(CatalogPitError):
    """Multiples snapshots validos cubren period_end."""


class SnapshotIntegrityError(CatalogPitError):
    """El sha256 del snapshot no coincide con el declarado."""


class ManifestError(CatalogPitError):
    """Manifest invalido (schema, campos, coherencia)."""


# --- Helpers ---

def _to_iso_date(v):
    """Normaliza cualquier representacion de fecha a ISO YYYY-MM-DD."""
    if v is None:
        return None
    ts = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date().isoformat()


def _sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_csv_path(catalog_root, version_id):
    return Path(catalog_root) / "catalog_snapshots" / SNAPSHOT_FILENAME_TEMPLATE.format(
        version_id=version_id
    )


def _snapshot_sha_path(catalog_root, version_id):
    return Path(catalog_root) / "catalog_snapshots" / SNAPSHOT_SHA_FILENAME_TEMPLATE.format(
        version_id=version_id
    )


def _manifest_path(catalog_root):
    return Path(catalog_root) / "catalog_manifest.json"

# --- API publica ---

def load_manifest(catalog_root):
    """Carga y valida estructuralmente el catalog_manifest.json.

    Raises:
      ManifestError si no existe, no es JSON valido, o schema incorrecto.
    """
    mp = _manifest_path(catalog_root)
    if not mp.exists():
        raise ManifestError("manifest no encontrado: {0}".format(mp))
    try:
        raw = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ManifestError("manifest ilegible: {0}".format(e))
    if not isinstance(raw, dict):
        raise ManifestError("manifest no es dict")
    if raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            "schema_version inesperado: {0}".format(raw.get("schema_version"))
        )
    snaps = raw.get("snapshots")
    if not isinstance(snaps, list):
        raise ManifestError("snapshots no es lista")
    return raw


def _interval_covers(valid_from, valid_to, period_end):
    """[valid_from, valid_to) contiene period_end."""
    p = _to_iso_date(period_end)
    vf = _to_iso_date(valid_from)
    vt = _to_iso_date(valid_to)
    if p is None or vf is None:
        return False
    if p < vf:
        return False
    if vt is not None and p >= vt:
        return False
    return True


def _select_snapshot_for_period(manifest, period_end):
    """Devuelve lista de entradas del manifest que cubren period_end."""
    hits = []
    for entry in manifest["snapshots"]:
        vf = entry.get("valid_from")
        vt = entry.get("valid_to")
        if _interval_covers(vf, vt, period_end):
            hits.append(entry)
    return hits


def verify_snapshot_integrity(version_id, *, catalog_root):
    """Verifica que sha256 recalculado == sha256 publicado.

    Verifica contra:
      1. El fichero snapshot_<vid>.sha256.
      2. La entrada del manifest para esa version_id.

    Raises:
      SnapshotIntegrityError si sha256 no coincide o falta evidencia.
    """
    csv_path = _snapshot_csv_path(catalog_root, version_id)
    sha_path = _snapshot_sha_path(catalog_root, version_id)

    if not csv_path.exists():
        raise SnapshotIntegrityError("csv no encontrado: {0}".format(csv_path))
    if not sha_path.exists():
        raise SnapshotIntegrityError("sha file no encontrado: {0}".format(sha_path))

    published_sha = sha_path.read_text(encoding="utf-8").strip()
    recalculated = _sha256_file(csv_path)

    if published_sha != recalculated:
        raise SnapshotIntegrityError(
            "SNAPSHOT MODIFICADO O CORRUPTO: "
            "hash recalculado != hash publicado"
        )

    manifest = load_manifest(catalog_root)
    entry = None
    for s in manifest["snapshots"]:
        if s.get("version_id") == version_id:
            entry = s
            break
    if entry is None:
        raise SnapshotIntegrityError(
            "version_id no presente en manifest: {0}".format(version_id)
        )
    if entry.get("sha256") != published_sha:
        raise SnapshotIntegrityError(
            "MANIFIESTO DESINCRONIZADO: "
            "sha256 manifest != sha256 publicado"
        )

    return True


def list_snapshots(*, catalog_root):
    """Devuelve la lista de entradas del manifest (copia superficial)."""
    manifest = load_manifest(catalog_root)
    return [dict(s) for s in manifest["snapshots"]]


def target_catalog_as_of(period_end, *, catalog_root):
    """Selecciona el snapshot que cubre period_end.

    Reglas:
      0 snapshots QUE CUBREN -> CatalogNotAvailable
      1 snapshot QUE CUBRE   -> (df, version_id, sha256)
      >1 snapshots QUE CUBREN -> CatalogAmbiguous

    Un snapshot existente que NO cubre period_end no convierte la
    consulta en valida.

    Tras seleccionar, verifica integridad del snapshot elegido.
    """
    manifest = load_manifest(catalog_root)
    hits = _select_snapshot_for_period(manifest, period_end)

    if len(hits) == 0:
        raise CatalogNotAvailable(
            "sin snapshot que cubra {0}".format(period_end)
        )
    if len(hits) > 1:
        raise CatalogAmbiguous(
            "multiples snapshots cubren {0}: {1}".format(
                period_end, [h.get("version_id") for h in hits]
            )
        )

    entry = hits[0]
    version_id = entry["version_id"]
    verify_snapshot_integrity(version_id, catalog_root=catalog_root)

    csv_path = _snapshot_csv_path(catalog_root, version_id)
    df = pd.read_csv(csv_path, dtype=str)
    return df, version_id, entry["sha256"]