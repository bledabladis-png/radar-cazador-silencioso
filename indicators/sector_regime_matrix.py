# -*- coding: utf-8 -*-
"""
Matriz de Régimen Sectorial v1.0
Sintetiza alineación descriptiva entre precio, amplitud, flujo y estructura.
No alimenta motores, scores, pesos ni State Machine.
"""
import pandas as pd
import numpy as np

from src.utils import _observation_date_from_df

from config.tickers import MARKET_TICKERS

SECTORS = MARKET_TICKERS['sectors']

def build_sector_regime_matrix(sector_breadth_df, sector_flow_df, sector_results, sector_wyckoff_distribution_df=None):
    """
    Combina datos existentes y devuelve DataFrame con la matriz.
    """
    if sector_breadth_df is None or sector_flow_df is None or sector_results is None:
        return pd.DataFrame()

    # C5.1 + Bloque 2 auditoria externa 2026-10-25: phase_map (fase dominante)
    # + structure_ratio CONTINUO (no binario) + n_estructuras. Formula:
    #   estructura_clara = acc + acc_forming + markup + dist + dist_forming + markdown
    #   structure_ratio  = (acc + acc_forming + markup) / estructura_clara
    # Null si estructura_clara == 0. NO es senal predictiva.
    phase_map = {}
    structure_ratio_map = {}
    n_estructuras_map = {}
    if sector_wyckoff_distribution_df is not None and not sector_wyckoff_distribution_df.empty:
        _dist = sector_wyckoff_distribution_df.copy()
        if 'date' in _dist.columns:
            try:
                _dist = _dist.sort_values('date').drop_duplicates('sector', keep='last')
            except (KeyError, ValueError, TypeError):
                pass
        _phase_cols = ('pct_accumulation', 'pct_markup', 'pct_range',
                       'pct_distribution', 'pct_markdown')
        _phase_names = {'pct_accumulation': 'ACCUMULATION', 'pct_markup': 'MARKUP',
                        'pct_range': 'RANGE', 'pct_distribution': 'DISTRIBUTION',
                        'pct_markdown': 'MARKDOWN'}
        for _, _row in _dist.iterrows():
            _sector = _row.get('sector')
            if _sector is None:
                continue
            # Bloque 2: structure_ratio continuo + n_estructuras
            try:
                _acc = float(_row.get('count_accumulation', 0) or 0)
                _accf = float(_row.get('count_accumulation_forming', 0) or 0)
                _mup = float(_row.get('count_markup', 0) or 0)
                _dst = float(_row.get('count_distribution', 0) or 0)
                _dstf = float(_row.get('count_distribution_forming', 0) or 0)
                _mdn = float(_row.get('count_markdown', 0) or 0)
                _alcista = _acc + _accf + _mup
                _clara = _alcista + _dst + _dstf + _mdn
                n_estructuras_map[_sector] = int(_clara)
                structure_ratio_map[_sector] = (_alcista / _clara) if _clara > 0 else None
            except (ValueError, TypeError):
                n_estructuras_map[_sector] = 0
                structure_ratio_map[_sector] = None
            # fase dominante
            _candidates = []
            for _col in _phase_cols:
                _v = _row.get(_col)
                if _v is not None and pd.notna(_v):
                    try:
                        _candidates.append((_phase_names[_col], float(_v)))
                    except (ValueError, TypeError):
                        pass
            phase_map[_sector] = max(_candidates, key=lambda x: x[1])[0] if _candidates else 'N/D'

    # Unificar por sector tomando la última fila si hay histórico
    breadth = sector_breadth_df.drop_duplicates(subset='sector', keep='last').set_index('sector')
    flow = sector_flow_df.drop_duplicates(subset='sector', keep='last').set_index('sector')

    rows = []
    for sector in SECTORS:
        if sector not in breadth.index or sector not in flow.index:
            continue

        price_ret = flow.loc[sector, 'price_ret_20d']
        pct_above = breadth.loc[sector, 'pct_above_ema50']
        flow_sum = flow.loc[sector, 'flow_20d_sum']
        phase = phase_map.get(sector, None)

        # Bloque 2: structure_pos retirada (era binaria, no robusta).
        # structure_ratio continua se publica como columna.
        # Validar solo las 3 condiciones que el sistema si produce robustas.
        if pd.isna(price_ret) or pd.isna(pct_above) or pd.isna(flow_sum):
            rows.append({
                'date': _observation_date_from_df(sector_breadth_df, col='date'),
                'sector': sector,
                'price_ret_20d': price_ret,
                'pct_above_ema50': pct_above,
                'flow_20d_sum': flow_sum,
                'wyckoff_phase': phase,
                'structure_ratio': (structure_ratio_map.get(sector)
                                     if sector_wyckoff_distribution_df is not None else None),
                'n_estructuras': (n_estructuras_map.get(sector, 0)
                                  if sector_wyckoff_distribution_df is not None else 0),
                'price_positive': np.nan,
                'breadth_positive': np.nan,
                'flow_positive': np.nan,
                'positive_conditions': np.nan,
                'data_complete': False,
                'regime_reading': 'N/D',
            })
            continue

        price_pos = price_ret > 0
        breadth_pos = pct_above > 50.0
        flow_pos = flow_sum > 0

        positives = sum([price_pos, breadth_pos, flow_pos])

        if positives == 3:
            lectura = 'Alineación positiva'
        elif positives == 2:
            lectura = 'Constructivo'
        elif positives == 1:
            lectura = 'Mixto'
        else:
            lectura = 'Debilidad alineada'

        rows.append({
            'date': _observation_date_from_df(sector_breadth_df, col='date'),
            'sector': sector,
            'price_ret_20d': price_ret,
            'pct_above_ema50': pct_above,
            'flow_20d_sum': flow_sum,
            'wyckoff_phase': phase,
            'structure_ratio': (structure_ratio_map.get(sector)
                                 if sector_wyckoff_distribution_df is not None else None),
            'n_estructuras': (n_estructuras_map.get(sector, 0)
                              if sector_wyckoff_distribution_df is not None else 0),
            'price_positive': price_pos,
            'breadth_positive': breadth_pos,
            'flow_positive': flow_pos,
            'positive_conditions': positives,
            'data_complete': True,
            'regime_reading': lectura,
        })

    return pd.DataFrame(rows)

