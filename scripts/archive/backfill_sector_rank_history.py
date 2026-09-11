# -*- coding: utf-8 -*-
"""
Backfill sector_rank_history para Punto 9.
"""
import pandas as pd
from pathlib import Path
from src.data_loader import download_market_data
from regimes.sector_regime import compute_sector_scores

def main():
    print("Descargando market...")
    df_market = download_market_data()
    if df_market is None or df_market.empty:
        print("Sin datos")
        return

    try:
        from src.utils import get_col
        bench = get_col(df_market, '^GSPC', 'Close').dropna()
    except Exception:
        bench = df_market.iloc[:,0].dropna()

    fechas = bench.index[-25:]
    print(f"Calculando ranking para {len(fechas)} fechas...")

    rows = []
    for fecha in fechas:
        df_cut = df_market.loc[:fecha]
        try:
            res = compute_sector_scores(df_cut)
            if res and 'ranking' in res and res['ranking']:
                for rank, (ticker, name, score, phase) in enumerate(res['ranking'], 1):
                    rows.append({
                        'date': pd.Timestamp(fecha).normalize(),
                        'sector': ticker,
                        'score': score,
                        'rank': rank,
                    })
        except Exception as e:
            print(f"Error en {fecha}: {e}")

    if rows:
        hist = pd.DataFrame(rows)
        hist = hist.drop_duplicates(subset=['date','sector'], keep='last').sort_values(['sector','date'])
        out = Path('outputs/history/sector_rank_history.csv')
        out.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(out, index=False)
        print(f"Guardado {len(hist)} filas en {out}")
        print("Ahora ejecuta manualmente update_rank_history para generar deltas, o vuelve a correr run.py")
    else:
        print("No se generaron filas")

if __name__ == '__main__':
    main()