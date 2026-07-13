from data.providers.finra import FinraProvider
finra = FinraProvider()
week = finra.get_latest_week()
print('1. Semana detectada:', week)
if week:
    print('2. Descargando T1...')
    data = finra.get_week_summary(week)
    print(f'   Filas T1: {len(data)}')
    if not data.empty:
        cols = data.columns.tolist()
        print(f'   Columnas: {cols}')
        sample = data[['issueSymbolIdentifier','totalWeeklyShareQuantity']].head(3)
        print('   Muestra:')
        print(sample.to_string())
    else:
        print('   No se descargaron datos T1.')
