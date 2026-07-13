from data.providers.finra import FinraProvider
import pandas as pd
from datetime import datetime, timedelta

finra = FinraProvider()
print('1. Proveedor disponible:', finra.is_available())
week = finra.get_latest_week()
print('2. Semana FINRA:', week)
if not week:
    print('   FALLO: No se encontro semana.')
    exit()

ats_data = finra.get_all_tiers(week)
print('3. Filas totales (todos los tiers):', len(ats_data))
if ats_data.empty:
    print('   FALLO: DataFrame vacio.')
    exit()

if 'issueSymbolIdentifier' not in ats_data.columns:
    print('   FALLO: Falta columna issueSymbolIdentifier. Columnas:', ats_data.columns.tolist())
    exit()

ats_vol = ats_data.groupby('issueSymbolIdentifier')['totalWeeklyShareQuantity'].sum().reset_index(name='ats_volume')
print('4. Simbolos unicos ATS:', len(ats_vol))

try:
    df_market = pd.read_csv('data/market_data.csv', header=[0,1], index_col=0, parse_dates=True)
    print('5. Yahoo Finance cargado.')
except Exception as e:
    print('   FALLO al cargar Yahoo:', e)
    exit()

end_date = pd.to_datetime(week) + timedelta(days=4)
week_data = df_market.loc[week:end_date.strftime('%Y-%m-%d')]
print('6. Dias en semana Yahoo:', len(week_data))
if week_data.empty:
    print('   FALLO: Sin datos en Yahoo para esa semana.')
    exit()

tickers = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLK', 'XLV', 'TLT', 'HYG', 'LQD', 'EEM', 'GLD', 'SLV', 'USO', 'UNG']
resultados = []
for t in tickers:
    try:
        vol_total = week_data[('Volume', t)].sum()
        row = ats_vol.loc[ats_vol['issueSymbolIdentifier'] == t, 'ats_volume']
        vol_ats = row.values[0] if len(row) > 0 else 0
        if vol_total > 0:
            dark_pool_pct = (vol_ats / vol_total) * 100
            resultados.append({'ticker': t, 'ats_volume': vol_ats, 'total_volume': vol_total, 'dark_pool_pct': dark_pool_pct})
    except KeyError:
        print(f'   Ticker {t} no encontrado en Yahoo Finance.')
    except Exception as e:
        print(f'   Error con {t}: {e}')

print('7. Resultados obtenidos:', len(resultados))
if not resultados:
    print('   FALLO: Ningun ticker coincidente.')
else:
    for r in resultados:
        print(f'   {r["ticker"]}: {r["dark_pool_pct"]:.2f}%')
