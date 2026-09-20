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


# --- Tests que documentan DIVERGENCIA (xfail hasta F2.4) ---


@pytest.mark.xfail(
    reason=(
        "Divergencia P60: _find_active_equivalence asume default TICKER "
        "si falta la columna identity_type. Contrato seccion 1.6 exige "
        "que el tipo venga declarado por la fuente, no asumido."
    )
)
def test_p60_csv_sin_identity_type_rechaza():
    """Contrato P60 seccion 1.6: si falta identity_type, la fila se rechaza.

    Estado actual: se asume TICKER silenciosamente.
    Cuando F2.4 autorice el fix, retirar el @pytest.mark.xfail.
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
    result = si._find_active_equivalence("037833100", "2026-03-31", eq_df)
    assert result == [], (
        "Esperado rechazo explicito, obtenido: {0}".format(result)
    )
