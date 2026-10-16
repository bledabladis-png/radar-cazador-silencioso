# -*- coding: utf-8 -*-
"""Test C3: contrato del return de get_etf_primary_flow_data.

Dictamen auditor externo 2026-10-16: el df in-memory debe llevar
primary_flow_z_raw para que el render pueda emitir el asterisco.
Auditoria 16-oct detecto que el CSV tenia la columna pero el df
in-memory no.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data.providers.ssga_fund_data as ssga


RETURN_COLS = {
    'ticker', 'Date', 'nav', 'shares_outstanding', 'total_net_assets',
    'primary_flow_usd', 'primary_flow_pct', 'primary_flow_z',
    'primary_flow_z_regime', 'primary_flow_z_raw',
}


def test_compute_primary_flow_produce_las_3_columnas_de_z():
    """_compute_primary_flow debe anadir _z, _regime y _raw."""
    df = pd.DataFrame({
        'Date': pd.date_range('2026-01-01', periods=30, freq='D'),
        'nav': [100.0] * 30,
        'shares_outstanding': [1_000_000 + i * 10_000 for i in range(30)],
        'total_net_assets': [100_000_000.0] * 30,
    })
    out = ssga._compute_primary_flow(df)
    for col in ('primary_flow_z', 'primary_flow_z_regime', 'primary_flow_z_raw'):
        assert col in out.columns, f"falta columna {col}"


def test_return_contract_incluye_z_raw():
    """El return de get_etf_primary_flow_data debe incluir las 3 columnas.
    Se inspecciona la firma estatica (no se ejecuta la descarga real)."""
    import inspect
    src = inspect.getsource(ssga.get_etf_primary_flow_data)
    # El return debe referenciar la lista con primary_flow_z_raw
    assert "primary_flow_z_raw" in src, (
        "get_etf_primary_flow_data no incluye primary_flow_z_raw en el return"
    )


def test_render_spdr_emite_asterisco_con_df_del_provider():
    """Integracion: el render recibe el df con _raw y emite asterisco."""
    from src.report.etf_flows import render_flujo_spdr
    df = pd.DataFrame([{
        'ticker': 'XLP', 'Date': '2026-10-14',
        'nav': 86.03, 'shares_outstanding': 166_771_809,
        'total_net_assets': 14_347_600_433,
        'primary_flow_usd': -709_758_464.25,
        'primary_flow_pct': -4.95,
        'primary_flow_z': -5.0,
        'primary_flow_z_regime': 'SATURATED',
        'primary_flow_z_raw': -6.787,
    }])
    out = "".join(render_flujo_spdr(df))
    assert "-5.00*" in out
    assert "(-6.79)" in out