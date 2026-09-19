"""Manifest de lineage SEC para el dataset 13F.

Version: 2.0 (2026-09-19)
FA-1.3 - manifest de integridad + lineage a 3 niveles.

Cadena auditable: SEC ZIP -> extraccion -> TSV -> normalizacion -> Parquet.

NO incluye: NIPC, canonical manager, amendments semanticos, CUSIP->issuer,
clasificacion institucional, agregaciones. Esos pasan por Gates posteriores.

Project root (P18/P26):
  DEFAULT_PROJECT_ROOT se resuelve por orden:
    1. Variable de entorno IAE_PROJECT_ROOT (si esta definida).
    2. Derivacion del paquete: parents[3] de este fichero = raiz del repo.
  NO se usa path absoluto hardcoded. El default es portatil.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .parser import PARSER_VERSION
from .schema import SCHEMA_VERSION

CHUNK_SIZE = 1024 * 1024

# P18/P26: project_root por env var, con fallback derivado del paquete.
# No path absoluto hardcoded: rompe portabilidad en CI / clones.
# Jerarquia: sec_13f/manifest.py -> sec_13f -> institutional_accumulation
#            -> src -> raiz del repo. parents[3] del fichero resuelto.
DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("IAE_PROJECT_ROOT")
    or Path(__file__).resolve().parents[3]
)


def _sha256_file(path):
    """SHA-256 de un fichero, leido en chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _relpath_or_abs(path, project_root):
    """Devuelve ruta relativa a project_root si es posible, sino absoluta."""
    try:
        return str(Path(path).resolve().relative_to(Path(project_root).resolve()).as_posix())
    except ValueError:
        return str(Path(path).resolve().as_posix())


def _file_metadata(path, project_root, kind="Fichero"):
    """Dict con file, sha256, size_bytes para un fichero.

    kind: etiqueta para el mensaje de error ("TSV" o "Parquet").
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(kind + " no existe: " + str(path))
    return {
        "file": _relpath_or_abs(path, project_root),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_manifest(
    *,
    quarter,
    source_period,
    source_url,
    source_filename,
    source_sha256,
    tsv_files,
    parquet_files,
    row_counts,
    validation,
    project_root=DEFAULT_PROJECT_ROOT,
    parser_version=PARSER_VERSION,
    schema_version=SCHEMA_VERSION,
):
    """Construye dict de manifest con lineage completo a 3 niveles.

    quarter:        "2026Q1" (identificador interno).
    source_period:  "01mar2026-31may2026" (nombre comercial SEC).
    tsv_files:      dict {nombre: Path TSV}.
    parquet_files:  dict {nombre: Path parquet}.
    row_counts:     dict {nombre: int filas}.
    validation:     dict con flags {expected_files_present, zip_crc_valid,
                    schema_valid}.
    """
    tsv_meta = {}
    for name, path in sorted(tsv_files.items()):
        m = _file_metadata(path, project_root, kind="TSV")
        m["rows"] = int(row_counts.get(name, 0))
        tsv_meta[name] = m

    parquet_meta = {}
    for name, path in sorted(parquet_files.items()):
        m = _file_metadata(path, project_root, kind="Parquet")
        m["rows"] = int(row_counts.get(name, 0))
        parquet_meta[name] = m

    return {
        "dataset": "SEC_FORM_13F",
        "schema_version": schema_version,
        "parser_version": parser_version,
        "quarter": quarter,
        "source_period": source_period,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "url": source_url,
            "filename": source_filename,
            "sha256": source_sha256,
        },
        "tsv_files": tsv_meta,
        "parquet_files": parquet_meta,
        "validation": {
            "expected_files_present": bool(validation.get("expected_files_present", False)),
            "zip_crc_valid": bool(validation.get("zip_crc_valid", False)),
            "schema_valid": bool(validation.get("schema_valid", False)),
        },
    }


def write_manifest(manifest, path):
    """Escribe manifest a JSON indentado. Escritura atomica via .tmp.

    Devuelve Path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return path


def read_manifest(path):
    """Lee un manifest JSON. Devuelve dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Manifest no existe: " + str(path))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)