"""Parser del dataset SEC 13F Data Set.

Version: 1.0 (2026-09-19)
FA-1.2 - lectura + dtypes + validacion estructural.

NO incluye:
  - Reglas de negocio (SH/PRN, DFND, dedup, CUSIP->ticker, NIPC).
  - Resolucion de reporting relationships.

Alcance: transformar 7 TSVs en dict {nombre: DataFrame} con dtypes
estables, validando contra schema.EXPECTED_COLUMNS.
"""
from pathlib import Path

import pandas as pd

from .schema import EXPECTED_COLUMNS, EXPECTED_FILES, validate_columns

PARSER_VERSION = "1.0"

STR_DTYPE = "string"
INT_DTYPE = "Int64"
FLOAT_DTYPE = "Float64"

# Columnas de fecha por TSV (formato DD-MMM-YYYY).
DATE_COLUMNS = {
    "SUBMISSION": ("FILING_DATE", "PERIODOFREPORT"),
    "COVERPAGE": ("DATEDENIEDEXPIRED", "DATEREPORTED"),
    "SIGNATURE": ("SIGNATUREDATE",),
}

# Columnas enteras nullable.
INT_COLUMNS = {
    "COVERPAGE": ("AMENDMENTNO",),
    "SUMMARYPAGE": ("OTHERINCLUDEDMANAGERSCOUNT", "TABLEENTRYTOTAL", "TABLEVALUETOTAL"),
    "OTHERMANAGER": ("OTHERMANAGER_SK",),
    "OTHERMANAGER2": ("SEQUENCENUMBER",),
    "INFOTABLE": (
        "INFOTABLE_SK",
        "VOTING_AUTH_SOLE",
        "VOTING_AUTH_SHARED",
        "VOTING_AUTH_NONE",
    ),
}

# Columnas reales nullable.
FLOAT_COLUMNS = {
    "SUBMISSION": (),
    "COVERPAGE": (),
    "SUMMARYPAGE": (),
    "OTHERMANAGER": (),
    "OTHERMANAGER2": (),
    "SIGNATURE": (),
    "INFOTABLE": ("VALUE", "SSHPRNAMT"),
}

def _build_dtype_map(tsv_name):
    """Devuelve dict {columna: "string"} para read_csv.

    TODAS las columnas se leen como string nullable, incluidas las
    numericas y de fecha. La conversion al dtype final se hace en
    _apply_dtypes con errors="coerce", de modo que un valor no
    parseable devuelve NaN en lugar de provocar ValueError critico
    en read_csv (escenario observado en tests FA-1.2).
    """
    cols = EXPECTED_COLUMNS[tsv_name]
    return {c: STR_DTYPE for c in cols}


def _apply_dtypes(df, tsv_name):
    """Convierte columnas al dtype final tras leer como string.

    - DATE: pd.to_datetime(errors="coerce").
    - INT:  pd.to_numeric(errors="coerce").astype("Int64").
    - FLOAT: pd.to_numeric(errors="coerce").astype("Float64").
    - Resto: queda como string nullable.

    Valor no parseable -> NaN (no imputa, no falla).
    """
    for col in DATE_COLUMNS.get(tsv_name, ()):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%d-%b-%Y", errors="coerce")
    for col in INT_COLUMNS.get(tsv_name, ()):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(INT_DTYPE)
    for col in FLOAT_COLUMNS.get(tsv_name, ()):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(FLOAT_DTYPE)
    return df

def parse_one(tsv_path, tsv_name, *, validate=True):
    """Lee un TSV y devuelve DataFrame con dtypes aplicados.

    tsv_path: Path al fichero.
    tsv_name: nombre sin extension (ej. "SUBMISSION").
    validate: si True, valida columnas contra schema.EXPECTED_COLUMNS.

    Lanza:
      FileNotFoundError si el fichero no existe.
      KeyError si tsv_name no esta en EXPECTED_COLUMNS.
      ValueError si validate=True y las columnas no coinciden.
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError("TSV no existe: " + str(tsv_path))
    if tsv_name not in EXPECTED_COLUMNS:
        raise KeyError("TSV no reconocido: " + tsv_name)

    dtype = _build_dtype_map(tsv_name)
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        dtype=dtype,
        keep_default_na=True,
        na_values=[""],
        encoding="utf-8",
    )

    if validate:
        errors = validate_columns(tsv_name, df.columns.tolist())
        if errors:
            raise ValueError(
                "Columnas de " + tsv_name + " no coinciden con schema: " + str(errors)
            )

    df = _apply_dtypes(df, tsv_name)
    return df

def parse_13f(extracted_dir, *, validate=True):
    """Lee los 7 TSVs de un directorio extraido.

    extracted_dir: Path al directorio con los 7 TSVs.
    validate: si True, valida columnas contra schema.

    Devuelve: dict {nombre_sin_extension: DataFrame}.
    Lanza FileNotFoundError si falta algun TSV esperado.
    """
    extracted_dir = Path(extracted_dir)
    if not extracted_dir.exists():
        raise FileNotFoundError("Directorio no existe: " + str(extracted_dir))

    result = {}
    for filename in EXPECTED_FILES:
        tsv_name = filename.replace(".tsv", "")
        tsv_path = extracted_dir / filename
        if not tsv_path.exists():
            raise FileNotFoundError("TSV faltante: " + str(tsv_path))
        result[tsv_name] = parse_one(tsv_path, tsv_name, validate=validate)
    return result