"""Orquestador de ingestion del dataset SEC 13F.

Version: 1.0 (2026-09-19)
FA-1.4 - integra downloader + parser + storage + manifest.

Cadena auditable:
  SEC ZIP -> extraccion -> TSV -> normalizacion -> Parquet -> manifest.

NO incluye: NIPC, reporting relationships, CUSIP->ticker, clasificacion.
Fuera de alcance hasta Gate FA-2.
"""
import hashlib
from pathlib import Path

from .downloader import (
    DEFAULT_USER_AGENT,
    _build_url,
    download_13f_zip,
    extract_13f_zip,
)
from .manifest import DEFAULT_PROJECT_ROOT, build_manifest, write_manifest
from .parser import parse_13f
from .schema import EXPECTED_FILES
from .storage import (
    DEFAULT_BASE_DIR,
    get_manifest_path,
    write_parquets,
)

CHUNK_SIZE = 1024 * 1024


def _sha256_file(path):
    """SHA-256 de un fichero (para el ZIP descargado)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()

def ingest_13f(
    quarter,
    source_period,
    base_dir=DEFAULT_BASE_DIR,
    user_agent=DEFAULT_USER_AGENT,
    force_download=False,
    project_root=DEFAULT_PROJECT_ROOT,
):
    """Ingesta completa del trimestre 13F.

    quarter:        identificador interno ("2026Q1").
    source_period:  nombre SEC ("01mar2026-31may2026").
    base_dir:       raiz de almacenamiento (default data/sec_13f).
    force_download: ignora cache del ZIP.
    project_root:   raiz para rutas relativas del manifest.

    Devuelve dict:
      {
        "quarter": str,
        "source_period": str,
        "zip_path": Path,
        "source_sha256": str,
        "tsv_paths": dict,
        "parquet_paths": dict,
        "row_counts": dict,
        "manifest_path": Path,
        "manifest": dict,
      }
    """
    base_dir = Path(base_dir)
    raw_dir = base_dir / "raw" / source_period
    extracted_dir = raw_dir / "extracted"

    # 1. URL
    source_url = _build_url(source_period)
    source_filename = source_period + "_form13f.zip"

    # 2. Descarga ZIP (con cache + retry)
    zip_path = download_13f_zip(
        source_period, raw_dir, user_agent=user_agent, force=force_download
    )
    source_sha256 = _sha256_file(zip_path)

    # 3. Extraccion (valida CRC + presencia de los 7 TSVs)
    tsv_paths = extract_13f_zip(zip_path, extracted_dir, force=force_download)

    # 4. Parseo (aplica dtypes + valida columnas)
    dfs = parse_13f(extracted_dir, validate=True)

    # 5. Escritura atomica de parquets
    parquet_meta = write_parquets(dfs, quarter, base_dir=base_dir)

    # 6. Row counts y mapa de paths
    row_counts = {name: len(df) for name, df in dfs.items()}
    parquet_paths = {name: meta["path"] for name, meta in parquet_meta.items()}

    # 7. Validacion (flags)
    validation = {
        "expected_files_present": len(tsv_paths) == len(EXPECTED_FILES),
        "zip_crc_valid": True,   # extract_13f_zip ya lo verifico
        "schema_valid": True,    # parse_13f validate=True ya lo verifico
    }

    # 8. Manifest
    manifest_dict = build_manifest(
        quarter=quarter,
        source_period=source_period,
        source_url=source_url,
        source_filename=source_filename,
        source_sha256=source_sha256,
        tsv_files=tsv_paths,
        parquet_files=parquet_paths,
        row_counts=row_counts,
        validation=validation,
        project_root=project_root,
    )
    manifest_path = write_manifest(
        manifest_dict, get_manifest_path(quarter, base_dir=base_dir)
    )

    return {
        "quarter": quarter,
        "source_period": source_period,
        "zip_path": zip_path,
        "source_sha256": source_sha256,
        "tsv_paths": tsv_paths,
        "parquet_paths": parquet_paths,
        "row_counts": row_counts,
        "manifest_path": manifest_path,
        "manifest": manifest_dict,
    }