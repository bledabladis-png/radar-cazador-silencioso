# -*- coding: utf-8 -*-
"""N3 (2026-10-13, auditor): contrato de WLS.

La validacion se extrajo a _validate_wls_inputs (funcion pura).
Valida:
- persistence_10d en [0, 1] (clip si fuera).
- Presencia de los 4 componentes (rs_z, flow_proxy_z_norm, rws_z, stab_z).
- Status por fila: OK | PERSISTENCE_OUT_OF_RANGE | INSUFFICIENT_COMPONENTS.
"""
import numpy as np
import pandas as pd

from indicators.stock_leader import _validate_wls_inputs, WLS_COMPONENTS


def _df(rs=0.5, flow=0.3, rws=0.2, stab=0.1, pers=0.7):
    return pd.DataFrame({
        'rs_z': [rs], 'flow_proxy_z_norm': [flow],
        'rws_z': [rws], 'stab_z': [stab],
        'persistence_10d': [pers],
    })


def test_n3_componentes_declarados():
    """El contrato declara los 4 componentes del WLS."""
    assert WLS_COMPONENTS == ('rs_z', 'flow_proxy_z_norm', 'rws_z', 'stab_z')


def test_n3_wls_persistence_en_rango_ok():
    pers, status, warns = _validate_wls_inputs(_df(pers=0.7))
    assert pers.iloc[0] == 0.7
    assert status.iloc[0] == 'OK'
    assert warns == []


def test_n3_wls_persistence_mayor_1_se_clipea():
    pers, status, warns = _validate_wls_inputs(_df(pers=70.0))
    assert pers.iloc[0] == 1.0
    assert status.iloc[0] == 'PERSISTENCE_OUT_OF_RANGE'
    assert len(warns) == 1
    assert '70' in warns[0]


def test_n3_wls_persistence_negativa_se_clipea():
    pers, status, warns = _validate_wls_inputs(_df(pers=-0.1))
    assert pers.iloc[0] == 0.0
    assert status.iloc[0] == 'PERSISTENCE_OUT_OF_RANGE'
    assert len(warns) == 1


def test_n3_wls_componente_ausente_nan():
    pers, status, warns = _validate_wls_inputs(_df(flow=np.nan))
    assert status.iloc[0] == 'INSUFFICIENT_COMPONENTS'


def test_n3_wls_persistence_nan_no_out_of_range():
    pers, status, warns = _validate_wls_inputs(_df(pers=np.nan))
    assert pd.isna(pers.iloc[0])
    assert status.iloc[0] == 'OK'


def test_n3_wls_componente_ausente_y_persistencia_out_of_range():
    """Si hay ambos problemas, INSUFFICIENT_COMPONENTS prevalece."""
    pers, status, warns = _validate_wls_inputs(_df(rs=np.nan, pers=70.0))
    assert status.iloc[0] == 'INSUFFICIENT_COMPONENTS'


def test_n3_wls_multiples_filas_estados_distintos():
    df = pd.DataFrame({
        'rs_z': [0.5, 0.5, 0.5],
        'flow_proxy_z_norm': [0.3, np.nan, 0.3],
        'rws_z': [0.2, 0.2, 0.2],
        'stab_z': [0.1, 0.1, 0.1],
        'persistence_10d': [0.7, 0.5, 70.0],
    })
    pers, status, warns = _validate_wls_inputs(df)
    assert status.iloc[0] == 'OK'
    assert status.iloc[1] == 'INSUFFICIENT_COMPONENTS'
    assert status.iloc[2] == 'PERSISTENCE_OUT_OF_RANGE'
    assert len(warns) == 1
