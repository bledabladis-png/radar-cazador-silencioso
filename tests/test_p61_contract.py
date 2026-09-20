"""Tests contractuales P61 (divergencia D1).

Documentan el estado del codigo frente al contrato
NIPC_CONTRATOS_SEMANTICOS_v1.md seccion 2.

Los tests marcados como @pytest.mark.xfail documentan una divergencia
conocida. Cuando F2.4 autorice el fix correspondiente, se retira el
marcador y el test debe pasar.

NO tocan codigo productivo.
"""
from unittest.mock import patch

import pytest

from src.institutional_accumulation import temporal_validity as tv
from src.institutional_accumulation.sec_13f.identity import (
    security_identity as si,
)


# --- Tests que HOY pasan (comportamiento contractual correcto) ---


def test_p61_resolve_source_status_etf_holdings_temp_unverified():
    """Contrato P61 seccion 2.6: etf_holdings sin vigencia -> TEMPORAL_UNVERIFIED."""
    result = tv.resolve_source_status(
        "etf_holdings", None, None, "2026-03-31"
    )
    assert result == tv.STATUS_TEMPORAL_UNVERIFIED


def test_p61_resolve_source_status_cusip_ticker_cubre_periodo():
    """Contrato P61 seccion 2.6: exceptions con vigencia -> VERIFIED si cubre."""
    result = tv.resolve_source_status(
        "cusip_ticker_exceptions", "2024-01-01", "2026-12-31", "2026-03-31"
    )
    assert result == tv.STATUS_VERIFIED


def test_p61_aggregate_status_6_reglas():
    """Contrato P61 seccion 2.3 + Q7: las 6 combinaciones de agregacion."""
    V = tv.STATUS_VERIFIED
    T = tv.STATUS_TEMPORAL_UNVERIFIED
    C = tv.STATUS_CONFLICT

    assert tv.aggregate_status([("X", V), ("X", V)]) == V
    assert tv.aggregate_status([("X", V), ("Y", V)]) == C
    assert tv.aggregate_status([("X", V), ("X", T)]) == V
    assert tv.aggregate_status([("X", V), ("Y", T)]) == C
    assert tv.aggregate_status([("X", T), ("X", T)]) == T
    assert tv.aggregate_status([("X", T), ("Y", T)]) == C


# --- Tests que documentan DIVERGENCIA (xfail hasta F2.4) ---


@pytest.mark.xfail(
    reason=(
        "Divergencia P61: resolve_security_identity no invoca "
        "temporal_validity.resolve_source_status. Usa dict hardcoded "
        "_SOURCE_TO_OP_STATUS. Contrato seccion 2.7."
    )
)
def test_p61_resolver_invoca_resolve_source_status():
    """Contrato P61 seccion 2.7: el resolver debe invocar resolve_source_status.

    Estado actual: no lo invoca. Usa _SOURCE_TO_OP_STATUS.
    Cuando F2.4 autorice el fix, retirar el @pytest.mark.xfail.
    """
    called = []
    original = tv.resolve_source_status

    def tracking(*args, **kwargs):
        called.append(args)
        return original(*args, **kwargs)

    with patch.object(tv, "resolve_source_status", tracking):
        si.resolve_security_identity("037833100", "2026-03-31")

    assert len(called) > 0, (
        "resolve_security_identity no invoca resolve_source_status"
    )


@pytest.mark.xfail(
    reason=(
        "Divergencia P61: cusip_ticker_exceptions con vigencia sale "
        "TEMPORAL_UNVERIFIED via crosswalk_internal, cuando el contrato "
        "exige VERIFIED si cubre el periodo."
    )
)
def test_p61_cusip_ticker_exceptions_verificado_end_to_end():
    """Contrato P61 seccion 2.6: exceptions con vigencia -> VERIFIED.

    Estado actual: cae en crosswalk_internal -> TEMPORAL_UNVERIFIED.
    Cuando F2.4 autorice el fix, retirar el @pytest.mark.xfail.
    """
    import pandas as pd

    cw_df = pd.DataFrame([
        {
            "CUSIP": "037833100",
            "ticker": "AAPL",
            "valid_from": pd.Timestamp("2024-01-01"),
            "valid_to": pd.Timestamp("2026-12-31"),
            "source": "cusip_ticker_exceptions",
        }
    ])
    result = si.resolve_security_identity(
        "037833100",
        "2026-03-31",
        crosswalk_internal_df=cw_df,
    )
    assert result["operational_mapping_status"] == "VERIFIED", (
        "Obtenido: {0}".format(result.get("operational_mapping_status"))
    )
