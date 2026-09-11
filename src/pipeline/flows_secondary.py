# -*- coding: utf-8 -*-
"""Fase 5 del pipeline: sintesis de flujo + N-PORT + QQQ Yahoo + QQQ NPORT-P.

Extraido de run.py (refactor C2, fase C2-6b).
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def compute_flows_secondary(sector_flow_rank, etf_primary_flow_data,
                             cftc_position_flow_data,
                             blackrock_dax_flow, blackrock_isf_flow,
                             amundi_lyxi_flow):
    """Carga sintesis de flujo + N-PORT + QQQ Yahoo + QQQ NPORT-P.

    Returns:
        dict con keys:
            flow_synthesis, nport_position_change_data,
            qqq_performance_data, qqq_nport_flow_data
    """
    # Sintesis descriptiva de flujo (sin superindicador)
    flow_synthesis = {}
    nport_position_change_data = None
    qqq_nport_flow_data = None
    qqq_performance_data = None
    try:
        # Direccion de flow_proxy (promedio de flow_proxy_z de lideres sectoriales)
        proxy_sign = np.nan
        if sector_flow_rank:
            proxy_sign = float(np.mean([f for _, f in sector_flow_rank]))
        flow_synthesis['flow_proxy_sign'] = proxy_sign

        # Direccion de ETF Primary Flow (promedio de primary_flow_z)
        primary_sign = np.nan
        if etf_primary_flow_data is not None and not etf_primary_flow_data.empty:
            primary_sign = float(etf_primary_flow_data['primary_flow_z'].mean())
        flow_synthesis['etf_primary_flow_sign'] = primary_sign

        # Direccion de CFTC Position Flow (promedio de flow_z)
        cftc_sign = np.nan
        if cftc_position_flow_data is not None and not cftc_position_flow_data.empty and 'flow_z' in cftc_position_flow_data.columns:
            cftc_sign = float(cftc_position_flow_data['flow_z'].mean())
        flow_synthesis['cftc_flow_sign'] = cftc_sign

        # Direccion de Europa Primary Flow (promedio de flow_zscore de DAXEX, ISF.L, LYXI)
        europe_sign = np.nan
        european_flows = []
        if blackrock_dax_flow is not None and not blackrock_dax_flow.empty and 'flow_zscore' in blackrock_dax_flow.columns:
            european_flows.append(float(blackrock_dax_flow['flow_zscore'].iloc[-1]))
        if blackrock_isf_flow is not None and not blackrock_isf_flow.empty and 'flow_zscore' in blackrock_isf_flow.columns:
            european_flows.append(float(blackrock_isf_flow['flow_zscore'].iloc[-1]))
        if amundi_lyxi_flow is not None and not amundi_lyxi_flow.empty and 'flow_zscore' in amundi_lyxi_flow.columns:
            european_flows.append(float(amundi_lyxi_flow['flow_zscore'].iloc[-1]))
        if european_flows:
            europe_sign = float(sum(european_flows) / len(european_flows))
        flow_synthesis['european_flow_sign'] = europe_sign

        # Cargar datos N-PORT mas recientes para el reporte
        try:
            nport_path = Path('outputs/history/sec_nport_position_change_quarterly.csv')
            if nport_path.exists():
                df_nport = pd.read_csv(nport_path, parse_dates=['REPORT_DATE'])
                if not df_nport.empty:
                    nport_position_change_data = df_nport.sort_values('REPORT_DATE', ascending=False).head(20)
        except Exception as e:
            print(f"  N-PORT no cargado: {e}")
            nport_position_change_data = None

        # Cargar flujo de participaciones QQQ NPORT-P
        try:
            qqq_nport_path = Path('outputs/history/qqq_nport_flow.csv')
            if qqq_nport_path.exists():
                qqq_nport_flow_data = pd.read_csv(qqq_nport_path)
        except Exception as e:
            print(f"  QQQ NPORT-P no cargado: {e}")
            qqq_nport_flow_data = None

        # Cargar rendimientos QQQ desde Yahoo Finance (si existe)
        try:
            perf_path = Path('outputs/history/qqq_returns_yahoo.csv')
            if perf_path.exists():
                mtime = datetime.fromtimestamp(perf_path.stat().st_mtime)
                age = datetime.now() - mtime
                if age <= timedelta(days=7):
                    qqq_performance_data = pd.read_csv(perf_path)
                else:
                    print(f"  QQQ returns Yahoo omitidos: datos con {age.days} dias")
            else:
                print("  [WARN] QQQ returns Yahoo no disponibles: no se encontro outputs/history/qqq_returns_yahoo.csv.")
        except Exception as e:
            print(f"  QQQ returns Yahoo no cargados: {e}")
            qqq_performance_data = None

        # Conteo de coincidencia de signos
        signs = []
        for s in [proxy_sign, primary_sign, cftc_sign, europe_sign]:
            if s > 0.1:
                signs.append(1)
            elif s < -0.1:
                signs.append(-1)
            else:
                signs.append(0)
        pos = sum(1 for x in signs if x > 0)
        neg = sum(1 for x in signs if x < 0)
        if pos == 3 or neg == 3:
            flow_synthesis['confidence'] = 'ALTA'
        elif pos == 2 or neg == 2:
            flow_synthesis['confidence'] = 'MEDIA'
        else:
            flow_synthesis['confidence'] = 'BAJA'
        print("  Sintesis de flujo calculada.")
    except Exception as e:
        print(f"  Sintesis de flujo omitida: {e}")
        flow_synthesis = {}

    return {
        'flow_synthesis': flow_synthesis,
        'nport_position_change_data': nport_position_change_data,
        'qqq_performance_data': qqq_performance_data,
        'qqq_nport_flow_data': qqq_nport_flow_data,
    }
