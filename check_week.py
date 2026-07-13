import pandas as pd
df = pd.read_csv('data/market_data.csv', header=[0,1], index_col=0, parse_dates=True)
# Ver fechas disponibles
print('Primera fecha:', df.index[0].date(), 'Última fecha:', df.index[-1].date())
# Ver si hay datos para la semana del 22 de junio (lunes)
try:
    week = df.loc['2026-06-22':'2026-06-26']
    print('Días en semana 22-jun:', len(week))
    # Buscar algunos tickers
    for t in ['XLF', 'SPY', 'AAPL', 'A']:
        if ('Volume', t) in week.columns:
            vol = week[('Volume', t)].sum()
            print(f'{t}: volumen total = {vol}')
        else:
            print(f'{t}: no encontrado en columnas')
except Exception as e:
    print('Error al acceder a la semana:', e)
