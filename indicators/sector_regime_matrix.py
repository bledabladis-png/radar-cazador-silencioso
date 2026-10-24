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

def _phase_positive(phase):
    if phase in ('ACCUMULATION', 'MARKUP'):
        return True
    elif phase == 'RANGE':
        return False  # neutral, no cuenta como positivo
    else:
        return False  # DISTRIBUTION, MARKDOWN, etc.

def build_sector_regime_matrix(sector_breadth_df, sector_flow_df, sector_results, sector_wyckoff_distribution_df=None):
    """
    Combina datos existentes y devuelve DataFrame con la matriz.
    """
    if sector_breadth_df is None or sector_flow_df is None or sector_results is None:
        return pd.DataFrame()

    # C5.1: fase y estructura desde holdings (sector_wyckoff_distribution_df).
    phase_map = {}
    structure_map = {}
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
            _acc = _row.get('pct_accumulation')
            _mkp = _row.get('pct_markup')
            try:
                _acc_f = float(_acc) if _acc is not None and pd.notna(_acc) else None
                _mkp_f = float(_mkp) if _mkp is not None and pd.notna(_mkp) else None
            except (ValueError, TypeError):
                _acc_f, _mkp_f = None, None
            if _acc_f is None and _mkp_f is None:
                structure_map[_sector] = None
            else:
                structure_map[_sector] = ((_acc_f or 0.0) + (_mkp_f or 0.0)) >= 50.0
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

        # C5.1: structure_pos desde holdings (o fallback a _phase_positive)
        if sector_wyckoff_distribution_df is not None:
            structure_pos = structure_map.get(sector)
        else:
            structure_pos = _phase_positive(phase) if phase else None

        # Validar que no falte ninguno
        if pd.isna(price_ret) or pd.isna(pct_above) or pd.isna(flow_sum) or structure_pos is None:
            rows.append({
                'date': _observation_date_from_df(sector_breadth_df, col='date'),
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
            'date': _observation_date_from_df(sector_breadth_df, col='date'),
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

