# -*- coding: utf-8 -*-
"""FIX-CFTC-FLOW-SIGN-ULTIMA-FECHA (2026-11-14): cftc_flow_sign usa ultima fecha.

Antes: mean() sobre todo el historico del CSV (7 semanas x 12 contratos)
publicado como "lectura actual". Ahora: mean de la ultima fecha.
Fallback a mean directo si el DataFrame no trae 'date' (compat fixture).
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.pipeline.flows_secondary import compute_flows_secondary


def _call(cftc_df):
    return compute_flows_secondary(
        sector_flow_rank=None,
        etf_primary_flow_data=None,
        cftc_position_flow_data=cftc_df,
        blackrock_dax_flow=None,
        blackrock_isf_flow=None,
        amundi_lyxi_flow=None,
        reference_date=datetime(2026, 11, 14, tzinfo=ZoneInfo("Europe/Madrid")),
    )


def test_cftc_usa_ultima_fecha_si_hay_date():
    df = pd.DataFrame({
        "date": ["2026-10-27"]*3 + ["2026-11-03"]*3,
        "flow_z": [-0.5, -0.5, -0.5,  0.9, 0.9, 0.9],
    })
    out = _call(df)
    assert out["flow_synthesis"]["cftc_flow_sign"] == pytest.approx(0.9)


def test_cftc_sin_date_fallback_mean_directo():
    # Fixture sin 'date' (compat test_pipeline_market_data_flows:161)
    df = pd.DataFrame({"flow_z": [0.2, 0.4, 0.6]})
    out = _call(df)
    assert out["flow_synthesis"]["cftc_flow_sign"] == pytest.approx(0.4)


def test_cftc_ultima_fecha_con_nan_mean_lo_ignora():
    df = pd.DataFrame({
        "date": ["2026-11-03"]*3,
        "flow_z": [0.5, float("nan"), 1.5],
    })
    out = _call(df)
    assert out["flow_synthesis"]["cftc_flow_sign"] == pytest.approx(1.0)


def test_cftc_vacio_na():
    out = _call(pd.DataFrame())
    assert pd.isna(out["flow_synthesis"]["cftc_flow_sign"])
