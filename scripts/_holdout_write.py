"""Escritura segura de parquets holdout (H-08 dictamen doc 76, T6).

Regla: una regeneracion normal del generador NO sobrescribe el
artefacto sellado. Para sustituirlo, se requiere --promote. Si el
hash semantico del nuevo contenido difiere del sellado, --promote
requiere adicionalmente --allow-hash-change.

El objetivo es que T6 pueda verificar "una ejecucion normal del
generador nunca reemplaza el fichero sellado": basta con invocar el
generador sin flags y comprobar que el sellado no cambia.

API:
    safe_write_parquet(df, out_path, promote=False, allow_hash_change=False)
    -> dict con keys: action, semantic_match, staging_path, message

Valores de action:
    'written'  primer sellado, o --promote aceptado
    'staged'   sin --promote: escrito a <out>.staged, sellado intacto
    'refused'  error (sellado ilegible, o --promote sin --allow-hash-change)
"""
import hashlib
from pathlib import Path

import pandas as pd


def _semantic_hash(df):
    """Mismo algoritmo que scripts/audit/holdout_manifest.py::_semantic_content_sha256."""
    if df is None or len(df) == 0:
        return None
    df_s = df.sort_index(axis=0).sort_index(axis=1)
    h = pd.util.hash_pandas_object(df_s, index=True)
    return hashlib.sha256(h.values.tobytes()).hexdigest()


def safe_write_parquet(df, out_path, *, promote=False, allow_hash_change=False):
    """Escribe df a out_path respetando la inmutabilidad del sellado."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = out_path.with_suffix(out_path.suffix + '.staged')

    # Caso 1: no existe sellado -> primer sellado, se escribe directamente
    if not out_path.exists():
        df.to_parquet(staging_path)
        staging_path.replace(out_path)
        return {
            "action": "written",
            "semantic_match": None,
            "staging_path": None,
            "message": "primer sellado escrito",
        }

    # Caso 2: sellado existe
    try:
        existing = pd.read_parquet(out_path)
    except Exception as e:
        return {
            "action": "refused",
            "semantic_match": None,
            "staging_path": None,
            "message": "no se puede leer el sellado: " + str(e),
        }

    h_new = _semantic_hash(df)
    h_old = _semantic_hash(existing)
    match = (h_new == h_old)

    if not promote:
        df.to_parquet(staging_path)
        if match:
            msg = ("contenido identico al sellado, no se toca. "
                   "Uso: --promote para sustituir el fichero (mismo contenido)")
        else:
            msg = ("contenido DIFIERE del sellado, staged en "
                   + str(staging_path)
                   + ". Uso: --promote --allow-hash-change para sustituir")
        return {
            "action": "staged",
            "semantic_match": match,
            "staging_path": str(staging_path),
            "message": msg,
        }

    # promote=True
    if not match and not allow_hash_change:
        df.to_parquet(staging_path)
        return {
            "action": "refused",
            "semantic_match": False,
            "staging_path": str(staging_path),
            "message": ("--promote requiere --allow-hash-change porque "
                        "el contenido difiere del sellado. Staged en "
                        + str(staging_path)),
        }

    df.to_parquet(staging_path)
    staging_path.replace(out_path)
    return {
        "action": "written",
        "semantic_match": match,
        "staging_path": None,
        "message": "sellado sustituido (--promote" + (
            " --allow-hash-change" if not match else "") + ")",
    }
