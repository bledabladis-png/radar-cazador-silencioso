"""Tests contractuales P60 (divergencia D3).

Documentan el estado del codigo frente al contrato
NIPC_CONTRATOS_SEMANTICOS_v1.md seccion 1.

Los tests marcados como @pytest.mark.xfail documentan una divergencia
conocida. Cuando F2.4 autorice el fix correspondiente, se retira el
marcador y el test debe pasar.

NO tocan codigo productivo.
"""
import pandas as pd
import pytest

from src.institutional_accumulation.sec_13f.identity import (
    security_identity as si,
)


# --- Tests que HOY pasan (comportamiento contractual correcto) ---


def test_p60_ticker_produce_equity_prefix():
    """Contrato P60 seccion 1.3: identity_type TICKER -> equity:<ticker>."""
    result = si._normalize_canonical("AAPL", "TICKER")
    assert result == "equity:AAPL"


def test_p60_figi_produce_figi_prefix():
    """Contrato P60 seccion 1.3: identity_type FIGI -> figi:<FIGI>."""
    result = si._normalize_canonical("BBG000B9XRY4", "FIGI")
    assert result == "figi:BBG000B9XRY4"


def test_p60_cusip_produce_null():
    """Contrato P60 seccion 1.3: identity_type CUSIP -> NULL (no canonical)."""
    result = si._normalize_canonical("037833100", "CUSIP")
    assert result is None


def test_p60_identity_type_none_raises():
    """Contrato P60 seccion 1.5: identity_type=None -> ValueError."""
    with pytest.raises(ValueError):
        si._normalize_canonical("AAPL", None)


def test_p60_identity_type_invalido_raises():
    """Contrato P60 seccion 1.5: identity_type invalido -> ValueError."""
    with pytest.raises(ValueError):
        si._normalize_canonical("AAPL", "INVALID")


# --- Tests de contrato P60 (fix aplicado tras F2.4) ---


def test_p60_csv_sin_identity_type_no_asume_ticker():
    """Contrato P60 seccion 1.6: no asumir tipo de identidad.

    El contrato exige que el tipo venga declarado por la fuente. No
    prescribe un comportamiento observable exacto (rechazo, lista
    vacia, UNRESOLVED, etc.): eso lo decide F2.4.

    Lo que SI prohibe el contrato es asumir TICKER silenciosamente.

    Estado actual: se asume TICKER. El test falla (xfail) porque
    obtenemos tuplas con prefijo "equity:" sin fuente declarada.
    """
    eq_df = pd.DataFrame([
        {
            "CUSIP_A": "037833100",
            "canonical_security": "AAPL",
            "valid_from": pd.Timestamp("2020-01-01"),
            "valid_to": None,
            # Sin columna identity_type
        }
    ])
    # P60 / F2.4: sin columna identity_type, fail-closed + warning.
    with pytest.warns(RuntimeWarning):
        result = si._find_active_equivalence("037833100", "2026-03-31", eq_df)
    assert result == [], (
        "Contrato P60: sin identity_type declarado NO se asume TICKER, "
        "se rechazan las filas. Obtenido: {0}".format(result)
    )

# --- C-06 (2026-09-29): S-03a y S-03b de security_identity ---
# Auditados como BUG LATENTE PRODUCTIVO. La cadena run.py -> iae_section
# -> pipeline_contractual.build_identities pasa el CSV por pd.read_csv
# (dtype=str). En cuanto cusip_equivalence.csv tenga filas reales (uso
# previsto: autoridad #1 de la cadena), el IAE cae con TypeError y el
# reporte diario pierde la seccion. Tambien: la invariante dura
# CANONICAL => canonical != None se violaba con identity_type=CUSIP.

def test_c06_crosswalk_valid_from_string_no_crash():
    """S-03a bis: defensa en profundidad en _find_active_crosswalk."""
    cw_raw = pd.DataFrame([
        {"CUSIP": "333333333", "ticker": "MSFT",
         "valid_from": "2020-01-01", "valid_to": None,
         "source": "cusip_ticker_exceptions"},
    ])
    assert cw_raw["valid_from"].dtype == object

    result = si.resolve_security_identity(
        "333333333", "2026-03-31",
        equivalence_df=None, crosswalk_internal_df=cw_raw,
    )
    assert result["security_resolution_status"] == "CANONICAL"
    assert result["canonical_security"] == "equity:MSFT"


def test_c06_identity_type_cusip_no_declara_canonical():
    """S-03b: invariante dura -- CANONICAL implica canonical != None.

    identity_type=CUSIP -> _normalize_canonical devuelve None.
    El status NO debe ser CANONICAL.
    """
    eq = pd.DataFrame([
        {"CUSIP_A": "111111111", "canonical_security": "111111111",
         "valid_from": pd.Timestamp("2020-01-01"), "valid_to": None,
         "source": "SEC", "reason": "test", "verified_by": "SEC",
         "source_document": None, "identity_type": "CUSIP"},
    ])
    result = si.resolve_security_identity(
        "111111111", "2026-03-31",
        equivalence_df=eq, crosswalk_internal_df=None,
    )
    assert result["security_resolution_status"] != "CANONICAL"
    assert result["canonical_security"] is None
    assert result["security_resolution_status"] == "OBSERVED_ONLY"

