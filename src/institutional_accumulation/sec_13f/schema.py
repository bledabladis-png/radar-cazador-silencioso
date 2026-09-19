"""Contrato declarativo del dataset SEC Form 13F Data Set.

Version: 1.0 (2026-09-19)
Basado en Gate 0 empirico sobre Q1 2026 (informe Gate 0).
NO contiene logica de negocio (parsing, dedup, CUSIP->ticker, etc.).

Verificado en D:/13f_q1_2026/extracted (7 TSVs, 3.3M filas INFOTABLE).
"""

SCHEMA_VERSION = "1.0"

EXPECTED_FILES = (
    "SUBMISSION.tsv",
    "COVERPAGE.tsv",
    "SUMMARYPAGE.tsv",
    "OTHERMANAGER.tsv",
    "OTHERMANAGER2.tsv",
    "SIGNATURE.tsv",
    "INFOTABLE.tsv",
)

EXPECTED_COLUMNS = {
    "SUBMISSION": (
        "ACCESSION_NUMBER",
        "FILING_DATE",
        "SUBMISSIONTYPE",
        "CIK",
        "PERIODOFREPORT",
    ),
    "COVERPAGE": (
        "ACCESSION_NUMBER",
        "REPORTCALENDARORQUARTER",
        "ISAMENDMENT",
        "AMENDMENTNO",
        "AMENDMENTTYPE",
        "CONFDENIEDEXPIRED",
        "DATEDENIEDEXPIRED",
        "DATEREPORTED",
        "REASONFORNONCONFIDENTIALITY",
        "FILINGMANAGER_NAME",
        "FILINGMANAGER_STREET1",
        "FILINGMANAGER_STREET2",
        "FILINGMANAGER_CITY",
        "FILINGMANAGER_STATEORCOUNTRY",
        "FILINGMANAGER_ZIPCODE",
        "REPORTTYPE",
        "FORM13FFILENUMBER",
        "CRDNUMBER",
        "SECFILENUMBER",
        "PROVIDEINFOFORINSTRUCTION5",
        "ADDITIONALINFORMATION",
    ),
    "SUMMARYPAGE": (
        "ACCESSION_NUMBER",
        "OTHERINCLUDEDMANAGERSCOUNT",
        "TABLEENTRYTOTAL",
        "TABLEVALUETOTAL",
        "ISCONFIDENTIALOMITTED",
    ),
    "OTHERMANAGER": (
        "ACCESSION_NUMBER",
        "OTHERMANAGER_SK",
        "CIK",
        "FORM13FFILENUMBER",
        "CRDNUMBER",
        "SECFILENUMBER",
        "NAME",
    ),
    "OTHERMANAGER2": (
        "ACCESSION_NUMBER",
        "SEQUENCENUMBER",
        "CIK",
        "FORM13FFILENUMBER",
        "CRDNUMBER",
        "SECFILENUMBER",
        "NAME",
    ),
    "SIGNATURE": (
        "ACCESSION_NUMBER",
        "NAME",
        "TITLE",
        "PHONE",
        "SIGNATURE",
        "CITY",
        "STATEORCOUNTRY",
        "SIGNATUREDATE",
    ),
    "INFOTABLE": (
        "ACCESSION_NUMBER",
        "INFOTABLE_SK",
        "NAMEOFISSUER",
        "TITLEOFCLASS",
        "CUSIP",
        "FIGI",
        "VALUE",
        "SSHPRNAMT",
        "SSHPRNAMTTYPE",
        "PUTCALL",
        "INVESTMENTDISCRETION",
        "OTHERMANAGER",
        "VOTING_AUTH_SOLE",
        "VOTING_AUTH_SHARED",
        "VOTING_AUTH_NONE",
    ),
}
# Claves primarias verificadas empiricamente en Gate 0 (100% unicidad).
PRIMARY_KEYS = {
    "SUBMISSION": ("ACCESSION_NUMBER",),
    "COVERPAGE": ("ACCESSION_NUMBER",),
    "SUMMARYPAGE": ("ACCESSION_NUMBER",),
    "OTHERMANAGER": ("ACCESSION_NUMBER", "OTHERMANAGER_SK"),
    "OTHERMANAGER2": ("ACCESSION_NUMBER", "SEQUENCENUMBER"),
    "SIGNATURE": ("ACCESSION_NUMBER",),
    "INFOTABLE": ("ACCESSION_NUMBER", "INFOTABLE_SK"),
}

# Formato SEC: DD-MMM-YYYY (ej. "31-MAR-2026").
DATE_COLUMNS = {
    "SUBMISSION": ("FILING_DATE", "PERIODOFREPORT"),
    "COVERPAGE": ("DATEDENIEDEXPIRED", "DATEREPORTED"),
    "SIGNATURE": ("SIGNATUREDATE",),
}

NUMERIC_COLUMNS = {
    "SUBMISSION": (),
    "COVERPAGE": ("AMENDMENTNO",),
    "SUMMARYPAGE": ("OTHERINCLUDEDMANAGERSCOUNT", "TABLEENTRYTOTAL", "TABLEVALUETOTAL"),
    "OTHERMANAGER": ("OTHERMANAGER_SK",),
    "OTHERMANAGER2": ("SEQUENCENUMBER",),
    "SIGNATURE": (),
    "INFOTABLE": (
        "INFOTABLE_SK", "VALUE", "SSHPRNAMT",
        "VOTING_AUTH_SOLE", "VOTING_AUTH_SHARED", "VOTING_AUTH_NONE",
    ),
}

# Categoricos con valores esperados (para validacion de dominio en fases posteriores).
CATEGORICAL_COLUMNS = {
    "SUBMISSION": {
        "SUBMISSIONTYPE": ("13F-HR", "13F-NT", "13F-HR/A", "13F-NT/A"),
    },
    "COVERPAGE": {
        "REPORTTYPE": ("13F HOLDINGS REPORT", "13F NOTICE", "13F COMBINATION REPORT"),
        "ISAMENDMENT": ("Y", "N", None),
    },
    "INFOTABLE": {
        "SSHPRNAMTTYPE": ("SH", "PRN"),
        "PUTCALL": ("Call", "Put", None),
        "INVESTMENTDISCRETION": ("SOLE", "DFND", "OTR"),
    },
}
def get_expected_columns(tsv_name):
    """Devuelve la tupla de columnas esperadas para un TSV.

    Acepta nombre con o sin extension .tsv.
    """
    key = tsv_name.replace(".tsv", "")
    if key not in EXPECTED_COLUMNS:
        raise KeyError("TSV no reconocido: " + tsv_name)
    return EXPECTED_COLUMNS[key]


def get_primary_key(tsv_name):
    """Devuelve la tupla de columnas que forman la clave primaria."""
    key = tsv_name.replace(".tsv", "")
    if key not in PRIMARY_KEYS:
        raise KeyError("TSV no reconocido: " + tsv_name)
    return PRIMARY_KEYS[key]


def validate_columns(tsv_name, actual_columns):
    """Valida que las columnas reales coinciden con las esperadas.

    Devuelve lista de errores (vacia si OK). No valida tipos ni contenido.
    """
    expected = set(get_expected_columns(tsv_name))
    actual = set(actual_columns)
    errors = []
    missing = expected - actual
    extra = actual - expected
    if missing:
        errors.append("Columnas faltantes: " + str(sorted(missing)))
    if extra:
        errors.append("Columnas inesperadas: " + str(sorted(extra)))
    return errors