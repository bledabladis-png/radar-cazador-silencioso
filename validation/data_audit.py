import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
from src.utils import get_col
from config.tickers import MARKET_TICKERS

print("Ejecutando auditoría de datos de mercado...")
tickers = []
for g in MARKET_TICKERS.values():
    if isinstance(g, dict): tickers.extend(g.values())
    elif isinstance(g, list): tickers.extend(g)
tickers = list(set(tickers))

# Descargar datos frescos
df = yf.download(tickers, period='10y', auto_adjust=True)
if not isinstance(df.columns, pd.MultiIndex):
    df.columns = pd.MultiIndex.from_tuples(df.columns)

print("\n=== AUDITORÍA DE DATOS ===")
print(f"Total de tickers: {len(tickers)}")
print(f"Fechas: {df.index[0].date()} a {df.index[-1].date()}")
print(f"Días totales: {len(df)}")

issues = []
for ticker in tickers:
    try:
        close = get_col(df, ticker, 'Close')
        missing = close.isna().sum()
        pct_missing = missing / len(close)
        
        # Outliers (z-score robusto > 5)
        ret = close.pct_change(fill_method=None).dropna()
        median = ret.median()
        mad = (ret - median).abs().median()
        z = ((ret - median) / (1.4826 * mad + 1e-9)).abs()
        outliers = (z > 5).sum()
        
        if pct_missing > 0.1 or outliers > 10:
            issues.append(f"  {ticker}: missing={pct_missing:.1%}, outliers={outliers}")
    except KeyError:
        issues.append(f"  {ticker}: no encontrado en datos")

duplicates = df.index.duplicated().sum()
print(f"Fechas duplicadas: {duplicates}")
print("Tickers con problemas:")
if issues:
    for i in issues:
        print(i)
else:
    print("  Ningún ticker presenta problemas significativos.")

# Estadísticas globales
print("\nResumen:")
print(f"  Tickers válidos: {len(tickers) - len(issues)} de {len(tickers)}")
print(f"  Cobertura temporal: {df.index[0].date()} a {df.index[-1].date()}")
