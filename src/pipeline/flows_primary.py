# -*- coding: utf-8 -*-
"""Fase 4 del pipeline: flujos primarios (SSGA + Sector Flow Char +
BlackRock + Amundi + QQQ SEC + CFTC).

Extraido de run.py (refactor C2, fase C2-6a).
"""

from pathlib import Path

import pandas as pd

from src.utils import append_dedup
from data.providers.retry_utils import retry_call
from data.providers.ssga_fund_data import get_etf_primary_flow_data
from data.providers.blackrock_fund_data import get_blackrock_dax_primary_flow
from data.providers.blackrock_isf_fund_data import get_blackrock_isf_primary_flow
from data.providers.blackrock_iwm_fund_data import get_blackrock_iwm_primary_flow
from data.providers.amundi_fund_data import get_amundi_lyxi_primary_flow
from data.providers.cftc_data import get_cftc_position_flow_data
from data.providers.qqq_sec_primary_flow import get_qqq_sec_primary_flow
from indicators.sector_flow_characteristics import compute_sector_flow_characteristics


def compute_flows_primary(df_market):
    """Calcula los flujos primarios (SSGA, BlackRock, Amundi, QQQ SEC, CFTC).

    Returns:
        dict con keys:
            etf_primary_flow_data, sector_flow_characteristics_df,
            blackrock_dax_flow, blackrock_isf_flow, blackrock_iwm_flow,
            amundi_lyxi_flow, qqq_sec_flow, cftc_position_flow_data
    """
    # Flujo primario ETF (SSGA)
    print("Calculando ETF Primary Flow (SSGA)...")
    try:
        etf_primary_flow_data = retry_call(get_etf_primary_flow_data)
        if etf_primary_flow_data is not None and not etf_primary_flow_data.empty:
            print("  ETF Primary Flow calculado.")
        else:
            etf_primary_flow_data = None
            print("  ETF Primary Flow sin datos.")
    except Exception as e:
        print(f"  ETF Primary Flow omitido: {e}")
        etf_primary_flow_data = None

    # --- Sector Flow Characteristics v1.0 (descriptivo) ---
    sector_flow_characteristics_df = None
    try:
        if etf_primary_flow_data is not None and not etf_primary_flow_data.empty:
            sector_flow_characteristics_df = compute_sector_flow_characteristics(
                'outputs/history/etf_primary_flow.csv', df_market
            )
            sfc_path = Path('outputs/history/sector_flow_characteristics.csv')
            sfc_path.parent.mkdir(parents=True, exist_ok=True)
            if not sector_flow_characteristics_df.empty:
                if sfc_path.exists():
                    hist_sfc = pd.read_csv(sfc_path)
                    sector_flow_characteristics_df = append_dedup(
                        hist_sfc, sector_flow_characteristics_df, ["date", "sector"]
                    )
                sector_flow_characteristics_df.to_csv(sfc_path, index=False, encoding='utf-8')
                print("  Sector Flow Characteristics calculado.")
    except Exception as e:
        print(f"  Sector Flow Characteristics omitido: {e}")
        sector_flow_characteristics_df = None

    # Flujo primario DAXEX (BlackRock)
    print("Calculando DAXEX Primary Flow (BlackRock)...")
    try:
        blackrock_dax_flow = retry_call(get_blackrock_dax_primary_flow)
        if blackrock_dax_flow is not None and not blackrock_dax_flow.empty:
            print("  DAXEX Primary Flow calculado.")
        else:
            blackrock_dax_flow = None
            print("  DAXEX Primary Flow sin datos.")
    except Exception as e:
        print(f"  DAXEX Primary Flow omitido: {e}")
        blackrock_dax_flow = None

    # Flujo primario ISF.L (BlackRock)
    print("Calculando ISF.L Primary Flow (BlackRock)...")
    try:
        blackrock_isf_flow = retry_call(get_blackrock_isf_primary_flow)
        if blackrock_isf_flow is not None and not blackrock_isf_flow.empty:
            print("  ISF.L Primary Flow calculado.")
        else:
            blackrock_isf_flow = None
            print("  ISF.L Primary Flow sin datos.")
    except Exception as e:
        print(f"  ISF.L Primary Flow omitido: {e}")
        blackrock_isf_flow = None

    # Flujo primario IWM (BlackRock)
    print("Calculando IWM Primary Flow (BlackRock)...")
    try:
        blackrock_iwm_flow = retry_call(get_blackrock_iwm_primary_flow)
        if blackrock_iwm_flow is not None and not blackrock_iwm_flow.empty:
            print("  IWM Primary Flow calculado.")
        else:
            blackrock_iwm_flow = None
            print("  IWM Primary Flow sin datos.")
    except Exception as e:
        print(f"  IWM Primary Flow omitido: {e}")
        blackrock_iwm_flow = None

    # Flujo primario LYXI (Amundi)
    print("Calculando LYXI Primary Flow (Amundi)...")
    try:
        amundi_lyxi_flow = retry_call(get_amundi_lyxi_primary_flow)
        if amundi_lyxi_flow is not None and not amundi_lyxi_flow.empty:
            print("  LYXI Primary Flow calculado.")
        else:
            amundi_lyxi_flow = None
            print("  LYXI Primary Flow sin datos.")
    except Exception as e:
        print(f"  LYXI Primary Flow omitido: {e}")
        amundi_lyxi_flow = None

    # Flujo primario QQQ (SEC, semestral/anual)
    print("Cargando QQQ SEC Primary Flow...")
    try:
        qqq_sec_flow = get_qqq_sec_primary_flow()
        if qqq_sec_flow is not None and not qqq_sec_flow.empty:
            print("  QQQ SEC Primary Flow cargado.")
        else:
            qqq_sec_flow = None
            print("  QQQ SEC Primary Flow sin datos.")
    except Exception as e:
        print(f"  QQQ SEC Primary Flow omitido: {e}")
        qqq_sec_flow = None

    # Posicionamiento CFTC (TFF, semanal)
    print("Calculando CFTC Position Flow (TFF)...")
    try:
        cftc_position_flow_data = retry_call(get_cftc_position_flow_data)
        if cftc_position_flow_data is not None and not cftc_position_flow_data.empty:
            print("  CFTC Position Flow calculado.")
        else:
            cftc_position_flow_data = None
            print("  CFTC Position Flow sin datos.")
    except Exception as e:
        print(f"  CFTC Position Flow omitido: {e}")
        cftc_position_flow_data = None

    return {
        'etf_primary_flow_data': etf_primary_flow_data,
        'sector_flow_characteristics_df': sector_flow_characteristics_df,
        'blackrock_dax_flow': blackrock_dax_flow,
        'blackrock_isf_flow': blackrock_isf_flow,
        'blackrock_iwm_flow': blackrock_iwm_flow,
        'amundi_lyxi_flow': amundi_lyxi_flow,
        'qqq_sec_flow': qqq_sec_flow,
        'cftc_position_flow_data': cftc_position_flow_data,
    }
