"""Merge de commodities (OilPriceAPI) en df_market (FU-021-3C-bis).

Los parquets commodities_*.parquet se producen por separado
(data/providers/futures.py, workflow update_futures.yml). Este modulo
solo LEE e integra en df_market.

Politica:
  - Solo merge en fechas comunes (no anade filas).
  - Sobrescribe columnas existentes; anade columnas nuevas si faltan.
  - Si los parquets no existen, no hace nada.
  - Idempotente: re-ejecutar no altera el resultado.

Refs: FU-021-3A (no filas hibridas), FU-021-3C-bis.
"""
from __future__ import annotations

import os

import pandas as pd


DEFAULT_FUTURES_PATH = 'data/commodities_futures.parquet'
DEFAULT_SPOT_PATH = 'data/commodities_spot.parquet'


def merge_commodities_into_market(
    df_market: pd.DataFrame,
    futures_path: str = DEFAULT_FUTURES_PATH,
    spot_path: str = DEFAULT_SPOT_PATH,
) -> pd.DataFrame:
    """Merge commodities parquet en df_market.

    Args:
        df_market: DataFrame con MultiIndex columns (field, ticker).
        futures_path: parquet de futuros (BZ=F, CL=F).
        spot_path: parquet de spot (GC=F, HG=F, NG=F).

    Returns:
        df_market modificado in-place (mismo objeto).
    """
    if df_market is None or df_market.empty:
        return df_market

    if not isinstance(df_market.columns, pd.MultiIndex):
        print('  [commodities_merge] df_market sin MultiIndex. Skip.')
        return df_market

    for path in (futures_path, spot_path):
        if not path or not os.path.exists(path):
            continue
        try:
            comm = pd.read_parquet(path)
        except Exception as e:
            print(f'  [commodities_merge][WARN] no se pudo leer {path}: {e}')
            continue
        if comm is None or comm.empty:
            continue
        if not isinstance(comm.columns, pd.MultiIndex):
            print(f'  [commodities_merge][WARN] {path} sin MultiIndex. Skip.')
            continue

        common_idx = df_market.index.intersection(comm.index)
        if len(common_idx) == 0:
            print(f'  [commodities_merge] {path}: sin fechas comunes '
                  f'(market={df_market.index.min().date()}..'
                  f'{df_market.index.max().date()}, '
                  f'comm={comm.index.min().date()}..'
                  f'{comm.index.max().date()})')
            continue

        for col in comm.columns:
            field, ticker = col
            values = comm.loc[common_idx, col]
            # K-FU-021-3D-03: cast defensivo a float64 antes de asignar.
            # commodities_spot.parquet puede traer columnas object
            # (Open/High/Low/Volume emitidos como None por FuturesProvider).
            # Pandas 2.x advierte al asignar object->float64; Pandas 3.x falla.
            # Dato no interpretable -> NaN (nunca imputar).
            values = pd.to_numeric(values, errors='coerce').astype('float64')
            if (field, ticker) in df_market.columns:
                df_market.loc[common_idx, (field, ticker)] = values.values
            else:
                new_col = pd.Series(index=df_market.index, dtype=float)
                new_col.loc[common_idx] = values.values
                df_market[(field, ticker)] = new_col
                df_market = df_market.sort_index(axis=1)

        n_cols = len(comm.columns)
        print(f'  [commodities_merge] {os.path.basename(path)}: '
              f'{n_cols} cols, {len(common_idx)} fechas comunes')

    return df_market
