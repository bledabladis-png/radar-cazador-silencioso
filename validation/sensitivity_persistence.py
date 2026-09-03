# -*- coding: utf-8 -*-
"""
Sensibilidad de Persistence a umbral y lookback usando historico real.

No optimiza parametros. Solo describe la variacion del indicador
ante cambios razonables de threshold y lookback.

Salida: outputs/audit/sensitivity_persistence.csv
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np
import yfinance as yf

from indicators.persistence import compute_persistence

OUTPUT_PATH = Path('outputs/audit/sensitivity_persistence.csv')
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SECTORS = ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
BENCHMARK = '^GSPC'

THRESHOLDS = [-0.02, 0.0, 0.02, 0.05]
LOOKBACKS = [4, 8, 12, 20]

def load_data() -> pd.DataFrame:
    """Descarga o carga datos de mercado desde cache local."""
    cache = Path('data/market_data.csv')
    if cache.exists():
        print(f"Usando cache: {cache}")
        df = pd.read_csv(cache, header=[0,1], index_col=0, parse_dates=True)
        # Comprobar que contiene los tickers necesarios
        try:
            df.loc[:, (slice(None), 'XLK')]
            return df
        except KeyError:
            print("Cache no contiene columnas esperadas. Descargando...")
    print("Descargando datos desde Yahoo Finance...")
    tickers = SECTORS + [BENCHMARK]
    data = yf.download(tickers, period='5y', auto_adjust=True, group_by='column', progress=False)
    # Reestructurar para compatibilidad con formato MultiIndex (Price, Ticker)
    data.columns = pd.MultiIndex.from_tuples([(col[0], col[1]) for col in data.columns])
    data.to_csv(cache)
    return data

def get_close(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Obtiene columna Close para un ticker."""
    if ('Close', ticker) in df.columns:
        return df[('Close', ticker)].squeeze()
    # si columns es simple
    if ticker in df.columns:
        return df[ticker]
    raise KeyError(f"No se encontro Close para {ticker}")

def main():
    data = load_data()
    close_spy = get_close(data, BENCHMARK).dropna()
    if close_spy.empty:
        raise RuntimeError("Sin datos del benchmark")

    # Calcular RS20 para cada sector
    rs20_data = {}
    for sector in SECTORS:
        try:
            close_sector = get_close(data, sector).dropna()
            common_idx = close_sector.index.intersection(close_spy.index)
            rs = close_sector.loc[common_idx] / close_spy.loc[common_idx]
            rs20 = rs.pct_change(20)
            rs20_data[sector] = rs20
        except Exception as e:
            print(f"  {sector}: {e}")
            rs20_data[sector] = pd.Series(dtype=float)

    rows = []
    for threshold in THRESHOLDS:
        for lookback in LOOKBACKS:
            valores = []
            for sector, series in rs20_data.items():
                p = compute_persistence(series, threshold=threshold, lookback=lookback)
                if p is not None:
                    valores.append(p)
            if valores:
                arr = np.array(valores)
                rows.append({
                    'threshold': threshold,
                    'lookback': lookback,
                    'media': float(np.mean(arr)),
                    'mediana': float(np.median(arr)),
                    'desviacion_std': float(np.std(arr)),
                    'minimo': float(np.min(arr)),
                    'maximo': float(np.max(arr)),
                    'n_sectores': int(len(arr)),
                })

    df_res = pd.DataFrame(rows)
    df_res.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"\nSensibilidad de Persistence guardada en {OUTPUT_PATH}")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
