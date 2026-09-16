"""Merge de CBOE (^VIX3M) en df_market (FU-021-3D).

El parquet data/cboe_vix3m.parquet se produce por separado
(data/providers/cboe_index.py, scripts/update_cboe.py, workflow
daily_run.yml). Este modulo solo LEE e integra en df_market.

Politica:
  - Solo merge en fechas comunes (no anade filas).
  - Sobrescribe la columna (Close|Open|High|Low|Volume, ^VIX3M) de
    Yahoo (que llega vacia) con el dato CBOE.
  - Si el parquet no existe, no hace nada (best-effort).
  - Idempotente: re-ejecutar no altera el resultado.

Refs: FU-021-3A (no filas hibridas), FU-021-3D, R6.
"""
from __future__ import annotations

import os

import pandas as pd


DEFAULT_CBOE_PATH = 'data/cboe_vix3m.parquet'


def merge_cboe_into_market(
    df_market: pd.DataFrame,
    cboe_path: str = DEFAULT_CBOE_PATH,
) -> pd.DataFrame:
    """Merge del parquet CBOE en df_market.

    Args:
        df_market: DataFrame con MultiIndex columns (field, ticker).
        cboe_path: parquet de ^VIX3M producido por CboeIndexProvider.

    Returns:
        df_market modificado in-place (mismo objeto).
    """
    if df_market is None or df_market.empty:
        return df_market

    if not isinstance(df_market.columns, pd.MultiIndex):
        print('  [cboe_merge] df_market sin MultiIndex. Skip.')
        return df_market

    if not cboe_path or not os.path.exists(cboe_path):
        return df_market

    try:
        cboe = pd.read_parquet(cboe_path)
    except Exception as e:
        print(f'  [cboe_merge][WARN] no se pudo leer {cboe_path}: {e}')
        return df_market

    if cboe is None or cboe.empty:
        return df_market

    if not isinstance(cboe.columns, pd.MultiIndex):
        print(f'  [cboe_merge][WARN] {cboe_path} sin MultiIndex. Skip.')
        return df_market

    common_idx = df_market.index.intersection(cboe.index)
    if len(common_idx) == 0:
        print(f'  [cboe_merge] {cboe_path}: sin fechas comunes '
              f'(market={df_market.index.min().date()}..'
              f'{df_market.index.max().date()}, '
              f'cboe={cboe.index.min().date()}..'
              f'{cboe.index.max().date()})')
        return df_market

    for col in cboe.columns:
        field, ticker = col
        values = cboe.loc[common_idx, col]
        if (field, ticker) in df_market.columns:
            df_market.loc[common_idx, (field, ticker)] = values.values
        else:
            new_col = pd.Series(index=df_market.index, dtype=float)
            new_col.loc[common_idx] = values.values
            df_market[(field, ticker)] = new_col
            df_market = df_market.sort_index(axis=1)

    print(f'  [cboe_merge] {os.path.basename(cboe_path)}: '
          f'{len(cboe.columns)} cols, {len(common_idx)} fechas comunes')

    return df_market