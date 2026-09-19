"""Persistencia de parquets 13F en el filesystem del Radar.

Version: 2.0 (2026-09-19)
FA-1.3 - escritura atomica + metadata por artefacto.

Estructura:
  data/sec_13f/processed/{quarter}/{TSV_NAME}.parquet   (7 ficheros)
  data/sec_13f/manifests/sec_13f_{quarter}.json         (1 manifest)

Escritura atomica: escribe a .tmp y hace os.replace.
"""
import hashlib
import os
from pathlib import Path

import pandas as pd

DEFAULT_BASE_DIR = Path("data/sec_13f")
CHUNK_SIZE = 1024 * 1024


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _processed_dir(base_dir, quarter):
    return Path(base_dir) / "processed" / quarter


def _raw_extracted_dir(base_dir, source_period):
    return Path(base_dir) / "raw" / source_period / "extracted"


def get_manifest_path(quarter, base_dir=DEFAULT_BASE_DIR):
    """Ruta esperada del manifest para un trimestre."""
    return Path(base_dir) / "manifests" / ("sec_13f_" + quarter + ".json")


def write_parquets(dfs, quarter, base_dir=DEFAULT_BASE_DIR, compression="snappy"):
    """Escribe cada DataFrame como parquet via .tmp + rename atomico.

    Devuelve: dict {nombre: {path, sha256, size_bytes, rows}}.
    """
    if not isinstance(dfs, dict) or not dfs:
        raise ValueError("dfs debe ser dict no vacio")

    out_dir = _processed_dir(base_dir, quarter)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for name, df in dfs.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError("dfs[" + name + "] no es DataFrame")
        final_path = out_dir / (name + ".parquet")
        tmp_path = final_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp_path, compression=compression, index=False)
        os.replace(tmp_path, final_path)
        result[name] = {
            "path": final_path,
            "sha256": _sha256_file(final_path),
            "size_bytes": final_path.stat().st_size,
            "rows": len(df),
        }
    return result


def load_parquets(quarter, base_dir=DEFAULT_BASE_DIR):
    """Carga los 7 parquets de un trimestre.

    Devuelve dict {nombre: DataFrame}.
    Falla si falta algun parquet esperado.
    """
    from .schema import EXPECTED_FILES
    out_dir = _processed_dir(base_dir, quarter)
    if not out_dir.exists():
        raise FileNotFoundError("Directorio processed no existe: " + str(out_dir))

    result = {}
    for filename in EXPECTED_FILES:
        name = filename.removesuffix(".tsv")
        parquet_path = out_dir / (name + ".parquet")
        if not parquet_path.exists():
            raise FileNotFoundError("Parquet faltante: " + str(parquet_path))
        result[name] = pd.read_parquet(parquet_path)
    return result