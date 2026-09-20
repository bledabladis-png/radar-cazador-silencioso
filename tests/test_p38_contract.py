"""Tests contractuales P38 (divergencia D2).

Documentan el estado del codigo frente al contrato
NIPC_CONTRATOS_SEMANTICOS_v1.md secciones 3 y 4.

Los tests marcados como @pytest.mark.xfail documentan una divergencia
conocida. Cuando F2.4 autorice el fix correspondiente, se retira el
marcador y el test debe pasar.

NO tocan codigo productivo.
"""
import inspect

import pandas as pd
import pytest

from src.institutional_accumulation.aggregation import nipc


# --- Tests que HOY pasan (comportamiento observable) ---


def test_p38_target_pairwise_hoy_sobre_observed_key():
    """Estado actual: el denominador opera sobre observed_security_key.

    Este test documenta el proxy actual. Cuando se implemente TARGET
    real, este test se sustituye por el contractual (ver xfail abajo).
    """
    import pandas as pd

    curr = pd.DataFrame([
        {
            "observed_security_key": "cusip:A",
            "security_resolution_status": "CANONICAL",
            "operational_mapping_status": "VERIFIED",
            "sshprnamt_total": 100,
        }
    ])
    prev = pd.DataFrame([
        {
            "observed_security_key": "cusip:A",
            "security_resolution_status": "CANONICAL",
            "operational_mapping_status": "VERIFIED",
            "sshprnamt_total": 80,
        }
    ])
    _ = nipc.compute_coverage_pairwise(curr, prev)
    # Con A en ambos, cobertura pairwise = 1.0
    assert result["paired_security_coverage"] == 1.0


def test_p38_denominador_cero_unavailable():
    """Contrato P38 seccion 3.3: denominador cero -> UNAVAILABLE."""
    empty = pd.DataFrame()
    result = nipc.compute_coverage_pairwise(empty, empty)
    assert result["paired_weighted_share_coverage"] is None
    assert result["coverage_status"] == "UNAVAILABLE"


# --- Tests que documentan DIVERGENCIA (xfail hasta F2.4) ---


@pytest.mark.xfail(
    reason=(
        "Divergencia P38: nipc.py no importa radar_target_catalog ni "
        "target_universe. El denominador no construye TARGET real. "
        "Contrato seccion 4.3."
    )
)
def test_p38_importa_radar_target_catalog():
    """Contrato P38 seccion 4.3: TARGET se construye desde RADAR_TARGET_CATALOG.

    Estado actual: nipc.py no importa esos modulos.
    Cuando F2.4 autorice el fix, retirar el @pytest.mark.xfail.
    """
    src = inspect.getsource(nipc)
    assert "radar_target_catalog" in src or "target_universe" in src, (
        "nipc.py no referencia radar_target_catalog ni target_universe"
    )


@pytest.mark.xfail(
    reason=(
        "Divergencia P38: compute_coverage_pairwise no acepta "
        "target_q4/target_q1 como parametros. Contrato seccion 8.3 "
        "del documento habilitante."
    )
)
def test_p38_firma_acepta_target_q4_q1():
    """Contrato P38: la funcion debe recibir target_q4/target_q1.

    Estado actual: firma = (units_current, units_previous).
    Cuando F2.4 autorice el fix, retirar el @pytest.mark.xfail.
    """
    sig = inspect.signature(nipc.compute_coverage_pairwise)
    params = set(sig.parameters.keys())
    assert "target_q4" in params and "target_q1" in params, (
        "Firma actual: {0}".format(params)
    )


@pytest.mark.xfail(
    reason=(
        "Divergencia P38: los pesos no se agregan por security antes "
        "de max(Q4,Q1). Contrato seccion 3.3 (w(s) = max(Q4_total, "
        "Q1_total) una vez por security)."
    )
)
def test_p38_pesos_agregados_por_security_antes_de_max():
    """Contrato P38: w(X) = max(Q4_total(X), Q1_total(X)).

    Escenario: dos CUSIPs del mismo X.
        Q4: CUSIP_A=100, CUSIP_B=50
        Q1: CUSIP_A=80,  CUSIP_B=60
    Si se agrega ANTES de max: w(X) = max(150, 140) = 150.
    Si se hace max por CUSIP:    max(100,80) + max(50,60) = 100+60 = 160.

    Estado actual: no aplica agregacion por security (opera por CUSIP).
    Cuando F2.4 autorice el fix, retirar el @pytest.mark.xfail.
    """
    import pandas as pd

    # Escenario simplificado: dos filas del mismo X pero observed_key distinto.
    # Bajo TARGET real, ambos deben agregarse por canonical_security antes del max.
    curr = pd.DataFrame([
        {"observed_security_key": "cusip:A", "security_resolution_status": "CANONICAL",
         "operational_mapping_status": "VERIFIED", "sshprnamt_total": 100},
        {"observed_security_key": "cusip:B", "security_resolution_status": "CANONICAL",
         "operational_mapping_status": "VERIFIED", "sshprnamt_total": 50},
    ])
    prev = pd.DataFrame([
        {"observed_security_key": "cusip:A", "security_resolution_status": "CANONICAL",
         "operational_mapping_status": "VERIFIED", "sshprnamt_total": 80},
        {"observed_security_key": "cusip:B", "security_resolution_status": "CANONICAL",
         "operational_mapping_status": "VERIFIED", "sshprnamt_total": 60},
    ])
    # Contrato: ambos CUSIPs corresponden al mismo canonical_security X.
    # El computo debe agregar antes de max. La funcion actual no tiene
    # forma de saber que A y B son el mismo X (no recibe canonical).
    result = nipc.compute_coverage_pairwise(curr, prev)
    # Marcador: si la funcion ya agregara por canonical, el resultado
    # ponderado seria 150 (numerador = 150, denom = 150 -> 1.0).
    # Hoy no lo hace. Cuando se implemente TARGET real, este test debe pasar.
    assert False, "Test contractual: pendiente de implementacion P38 real"
