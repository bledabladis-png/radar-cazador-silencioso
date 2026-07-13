import pandas as pd
from datetime import datetime, timedelta
from data.providers.finra import FinraProvider

finra = FinraProvider()
week = finra.get_latest_week()
print('1. Semana FINRA:', week)

# Cargar datos de mercado
df_market = pd.read_csv('data/market_data.csv', header=[0,1], index_col=0, parse_dates=True)
print('2. Fechas Yahoo:', df_market.index[0].date(), 'a', df_market.index[-1].date())

# Verificar si hay datos para esa semana
end_date = pd.to_datetime(week) + timedelta(days=4)
print(f'3. Buscando datos desde {week} hasta {end_date.strftime("%Y-%m-%d")}')
try:
    week_data = df_market.loc[week:end_date.strftime('%Y-%m-%d')]
    print(f'   Dias encontrados: {len(week_data)}')
    
    # Verificar si los tickers existen
    for t in ['XLF', 'SPY', 'QQQ', 'IWM']:
        vol_col = ('Volume', t)
        if vol_col in week_data.columns:
            vol = week_data[vol_col].sum()
            print(f'   {t}: volumen = {vol}')
        else:
            print(f'   {t}: NO ENCONTRADO en columnas')
            # Mostrar columnas disponibles con ese ticker
            cols_with_ticker = [c for c in week_data.columns if t in str(c)]
            print(f'   Columnas con {t}: {cols_with_ticker[:5]}')
except Exception as e:
    print(f'   ERROR: {e}')
