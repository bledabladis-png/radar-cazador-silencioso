"""Tests contractuales P38 (divergencia D2).

Documentan el estado del codigo frente al contrato
NIPC_CONTRATOS_SEMANTICOS_v1.md secciones 3 y 4.

Los tests marcados con @pytest.mark.xfail documentan una divergencia
conocida. Cuando F2.4 autorice el fix correspondiente, se retira el
marcador y el test debe pasar.

API normativa P38 (a dictamen F2.4):

    compute_contractual_coverage(...)   FUNCION CONTRACTUAL
    compute_nipc(...)                   orquestador
    compute_coverage_pairwise(...)      legacy / proxy (deprecada)

`nipc.py` NO importa `radar_target_catalog` ni `target_universe`. El
TARGET se construye externamente (identity/target_builder.py) y se pasa
como parametro. Esa es la arquitectura contractual.

NO tocan codigo productivo.
"""
import inspect

import pandas as pd
import pytest

from src.institutional_accumulation.aggregation import nipc


# --- Tests que HOY pasan (comportamiento observable) ---


def test_p38_legacy_denominador_cero_unavailable():
    """Comportamiento legacy/proxy: denominador cero -> UNAVAILABLE.

    Este test verifica el comportamiento de la API LEGACY
    (compute_coverage_pairwise), no de la API contractual P38
    (compute_contractual_coverage, a implementar en A.6).
    Se mantiene como regresion. La evidencia contractual vendra del
    test contra compute_contractual_coverage.
    """
    empty = pd.DataFrame()
    result = nipc.compute_coverage_pairwise(empty, empty)
    assert result["paired_weighted_share_coverage"] is None
    assert result["coverage_status"] == "UNAVAILABLE"


def test_p38_legacy_usa_observed_key():
    """Documenta el comportamiento actual: la funcion legacy opera
    sobre observed_security_key (proxy). Cuando se implemente TARGET
    real (A.6.2-P38-materialize), este test se sustituye por el
    contractual."""
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
    result = nipc.compute_coverage_pairwise(curr, prev)
    assert result["paired_security_coverage"] == 1.0


# --- Tests que documentan DIVERGENCIA (xfail hasta F2.4) ---


@pytest.mark.xfail(
    reason=(
        "Divergencia P38: compute_contractual_coverage no existe como "
        "API normativa. La arquitectura exige que TARGET se reciba como "
        "parametro externo (construido por target_builder). Contrato "
        "seccion 3.2 y 4.3."
    )
)
def test_p38_existe_compute_contractual_coverage():
    """Contrato P38: la funcion contractual debe existir.

    Estado actual: no existe. Cuando A.6.2-P38 la implemente, este
    test pasa.
    """
    from src.institutional_accumulation.aggregation import coverage
    assert hasattr(coverage, "compute_contractual_coverage"), (
        "Falta aggregation/coverage.py::compute_contractual_coverage"
    )


@pytest.mark.xfail(
    reason=(
        "Divergencia P38: compute_contractual_coverage no recibe "
        "target_q4/target_q1 como parametros externos. Contrato 8.3."
    )
)
def test_p38_contractual_recibe_target_externo():
    """Contrato P38: TARGET se recibe, no se construye internamente.

    Estado actual: la funcion no existe. Cuando exista, su firma debe
    incluir target_q4/target_q1 como parametros.
    """
    from src.institutional_accumulation.aggregation import coverage
    if not hasattr(coverage, "compute_contractual_coverage"):
        pytest.fail("compute_contractual_coverage no existe todavia")
    sig = inspect.signature(coverage.compute_contractual_coverage)
    params = set(sig.parameters.keys())
    assert "target_q4" in params and "target_q1" in params, (
        "Firma actual: {0}".format(params)
    )


@pytest.mark.xfail(
    reason=(
        "Divergencia P38: la agregacion de pesos por shareClassFIGI "
        "antes de max(Q4,Q1) no esta implementada. Contrato 3.3."
    )
)
def test_p38_agrega_pesos_por_shareclass_figi():
    """Contrato P38 seccion 3.3: w(X) = max(Q4_total(X), Q1_total(X)).

    Escenario (bajo Modelo A - shareClassFIGI):
        Q4: CUSIP_A=100, CUSIP_B=50  (ambos FIGI_X)
        Q1: CUSIP_A=80,  CUSIP_B=60  (ambos FIGI_X)

    Agregacion correcta: w(FIGI_X) = max(150, 140) = 150.
    Agregacion incorrecta: max(100,80) + max(50,60) = 160.

    La regla contractual exige agregar antes de max.
    """
    from src.institutional_accumulation.aggregation import coverage
    if not hasattr(coverage, "compute_contractual_coverage"):
        pytest.fail("compute_contractual_coverage no existe todavia")

    # Este test se activara cuando exista la funcion contractual.
    # Requiere: (a) TARGET_Q4/TARGET_Q1 como sets de shareClassFIGI,
    # (b) weights_q4/weights_q1 agregados por shareClassFIGI.
    pytest.fail(
        "Test contractual pendiente: requiere compute_contractual_coverage "
        "+ agregacion por shareClassFIGI implementada"
    )
