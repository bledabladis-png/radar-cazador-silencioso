# Auditoría Maestra - Descarga incremental de histórico de acciones
# Guarda en data/stock_prices_historical.csv (formato MultiIndex).
# En ejecuciones futuras, solo descarga los días faltantes.
# Respeta las premisas: no trading bot, no sobreingeniería, determinista.

import pandas as pd
import numpy as np
import yfinance as yf
import os, time
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential

HOLDINGS_FILE = "data/etf_holdings.csv"
OUTPUT_FILE = "data/stock_prices_historical.csv"
BATCH_SIZE = 5
DELAY_SECONDS = 2
YEARS_BACK = 5  # Historial máximo si no existe archivo previo
MAX_RETRIES = 3

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_batch(tickers, start_date, end_date):
    """Descarga un lote de tickers desde Yahoo Finance."""
    data = yf.download(
        tickers=" ".join(tickers),
        start=start_date,
        end=end_date,
        progress=False,
        group_by="ticker",
        auto_adjust=True,
        threads=False
    )
    return data

def build_multiindex_df(raw_data, tickers):
    """Convierte el resultado de yfinance a MultiIndex (Price, ticker)."""
    frames = []
    for t in tickers:
        try:
            if len(tickers) == 1:
                df_t = raw_data.copy()
            else:
                df_t = raw_data[t].copy()
            df_t.columns = pd.MultiIndex.from_product([df_t.columns, [t]])
            frames.append(df_t)
        except:
            pass
    if frames:
        return pd.concat(frames, axis=1).sort_index(axis=1)
    return None

print("=" * 70)
print("ACTUALIZACIÓN INCREMENTAL DE HISTÓRICO DE ACCIONES")
print("=" * 70)

# 1. Leer tickers
print("\n[1/3] Leyendo tickers desde etf_holdings.csv ...")
holdings = pd.read_csv(HOLDINGS_FILE)
tickers_all = sorted(holdings["ticker"].unique())
print(f"  Total de tickers: {len(tickers_all)}")

# 2. Determinar rango de fechas (incremental)
print("[2/3] Determinando rango de fechas ...")
if os.path.exists(OUTPUT_FILE):
    existing = pd.read_csv(OUTPUT_FILE, header=[0,1], index_col=0, parse_dates=True)
    last_date = existing.index[-1]
    print(f"  Archivo existente. Última fecha: {last_date.date()}")
    start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    if datetime.strptime(start_date, "%Y-%m-%d") > datetime.today():
        print("  Ya está actualizado. No hay datos nuevos que descargar.")
        exit()
else:
    start_date = (datetime.today() - timedelta(days=YEARS_BACK*365)).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")
    existing = None
print(f"  Descargando desde {start_date} hasta {end_date}")

# 3. Descargar en lotes
print("[3/3] Descargando datos (puede tardar varios minutos) ...")
batches = [tickers_all[i:i+BATCH_SIZE] for i in range(0, len(tickers_all), BATCH_SIZE)]
all_frames = []
errors = []

for i, batch in enumerate(batches):
    try:
        raw = download_batch(batch, start_date, end_date)
        df_batch = build_multiindex_df(raw, batch)
        if df_batch is not None:
            all_frames.append(df_batch)
        time.sleep(DELAY_SECONDS)
        if (i+1) % 10 == 0:
            print(f"  Lote {i+1}/{len(batches)} completado.")
    except Exception as e:
        errors.append((batch, str(e)))
        print(f"  Error en lote {batch}: {e}")

if not all_frames:
    print("ERROR: No se descargaron datos.")
    exit()

# 4. Combinar y guardar
new_data = pd.concat(all_frames, axis=1).sort_index(axis=1)
if existing is not None:
    combined = pd.concat([existing, new_data]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
else:
    combined = new_data

combined.to_csv(OUTPUT_FILE)
print(f"\nArchivo guardado: {OUTPUT_FILE}")
print(f"  Fechas: {combined.index[0].date()} a {combined.index[-1].date()}")
print(f"  Acciones: {len(combined.columns.levels[1]) if hasattr(combined.columns, 'levels') else len(combined.columns)//5}")
print("=" * 70)
