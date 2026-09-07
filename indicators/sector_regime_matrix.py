# -*- coding: utf-8 -*-
"""
Matriz de Régimen Sectorial v1.0
Sintetiza alineación descriptiva entre precio, amplitud, flujo y estructura.
No alimenta motores, scores, pesos ni State Machine.
"""
import pandas as pd
import numpy as np

SECTORS = ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']

def _phase_positive(phase):
    if phase in ('ACCUMULATION', 'MARKUP'):
        return True
    elif phase == 'RANGE':
        return False  # neutral, no cuenta como positivo
    else:
        return False  # DISTRIBUTION, MARKDOWN, etc.

def build_sector_regime_matrix(sector_breadth_df, sector_flow_df, sector_results):
    """
    Combina datos existentes y devuelve DataFrame con la matriz.
    """
    if sector_breadth_df is None or sector_flow_df is None or sector_results is None:
        return pd.DataFrame()

    # Mapear fase Wyckoff desde sector_results['ranking']
    phase_map = {}
    if 'ranking' in sector_results and sector_results['ranking']:
        for item in sector_results['ranking']:
            if len(item) >= 4:
                ticker, name, score, phase = item[0], item[1], item[2], item[3]
                phase_map[ticker] = phase

    # Unificar por sector
    breadth = sector_breadth_df.set_index('sector')
    flow = sector_flow_df.set_index('sector')

    rows = []
    for sector in SECTORS:
        if sector not in breadth.index or sector not in flow.index:
            continue

        price_ret = flow.loc[sector, 'price_ret_20d']
        pct_above = breadth.loc[sector, 'pct_above_ema50']
        flow_sum = flow.loc[sector, 'flow_20d_sum']
        phase = phase_map.get(sector, None)

        # Validar que no falte ninguno
        if pd.isna(price_ret) or pd.isna(pct_above) or pd.isna(flow_sum) or phase is None:
            rows.append({
                'date': pd.Timestamp.now().normalize(),
                'sector': sector,
                'price_ret_20d': price_ret,
                'pct_above_ema50': pct_above,
                'flow_20d_sum': flow_sum,
                'wyckoff_phase': phase,
                'price_positive': np.nan,
                'breadth_positive': np.nan,
                'flow_positive': np.nan,
                'structure_positive': np.nan,
                'positive_conditions': np.nan,
                'data_complete': False,
                'regime_reading': 'N/D',
            })
            continue

        price_pos = price_ret > 0
        breadth_pos = pct_above > 50.0
        flow_pos = flow_sum > 0
        structure_pos = _phase_positive(phase)

        positives = sum([price_pos, breadth_pos, flow_pos, structure_pos])

        if positives == 4:
            lectura = 'Alineación positiva'
        elif positives == 3:
            lectura = 'Constructivo'
        elif positives == 2:
            lectura = 'Mixto'
        elif positives == 1:
            lectura = 'Débil'
        else:
            lectura = 'Debilidad alineada'

        rows.append({
            'date': pd.Timestamp.now().normalize(),
            'sector': sector,
            'price_ret_20d': price_ret,
            'pct_above_ema50': pct_above,
            'flow_20d_sum': flow_sum,
            'wyckoff_phase': phase,
            'price_positive': price_pos,
            'breadth_positive': breadth_pos,
            'flow_positive': flow_pos,
            'structure_positive': structure_pos,
            'positive_conditions': positives,
            'data_complete': True,
            'regime_reading': lectura,
        })

    return pd.DataFrame(rows)
