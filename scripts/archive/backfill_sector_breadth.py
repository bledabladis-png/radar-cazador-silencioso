# -*- coding: utf-8 -*-
"""
Backfill sector_breadth histórico para Punto 14.
Calcula la amplitud sectorial para los últimos 90 días y regenera momentum.
"""
import pandas as pd
from pathlib import Path
from src.data_loader import download_market_data
from src.stock_data_loader import download_stock_prices
from indicators.sector_breadth import compute_sector_breadth
from indicators.sector_breadth_momentum import compute_sector_breadth_momentum

def main():
    print("Descargando market...")
    df_market = download_market_data()
    print("Descargando stocks...")
    df_stocks = download_stock_prices()
    holdings = pd.read_csv('data/etf_holdings.csv')

    if df_market is None or df_stocks is None or df_stocks.empty:
        print("Sin datos")
        return

    # Fechas de cierre comunes: usar el índice de df_market si es DatetimeIndex
    if df_market is not None and len(df_market.index) > 0:
        fechas = pd.DatetimeIndex(df_market.index)[-90:]
    elif df_stocks is not None and len(df_stocks.index) > 0:
        fechas = pd.DatetimeIndex(df_stocks.index)[-90:]
    else:
        fechas = []
    print(f"Calculando sector_breadth para {len(fechas)} fechas...")

    frames = []
    for fecha in fechas:
        try:
            df_breadth = compute_sector_breadth(df_market, df_stocks, holdings, as_of_date=fecha)
            if not df_breadth.empty:
                df_breadth['date'] = pd.Timestamp(fecha).normalize()
                frames.append(df_breadth)
        except Exception as e:
            print(f"Error en {fecha}: {e}")

    if frames:
        hist = pd.concat(frames, ignore_index=True)
        hist = hist.drop_duplicates(subset=['date','sector'], keep='last').sort_values(['sector','date'])
        out_path = Path('outputs/history/sector_breadth.csv')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(out_path, index=False)
        print(f"Guardado {len(hist)} filas en {out_path}")
    else:
        print("No se generó histórico")

    # Regenerar momentum
    try:
        mom = compute_sector_breadth_momentum('outputs/history/sector_breadth.csv')
        mom_path = Path('outputs/history/sector_breadth_momentum.csv')
        mom_path.parent.mkdir(parents=True, exist_ok=True)
        mom.to_csv(mom_path, index=False)
        print(f"Momentum regenerado: {len(mom)} filas")
    except Exception as e:
        print(f"Error momentum: {e}")

if __name__ == '__main__':
    main()