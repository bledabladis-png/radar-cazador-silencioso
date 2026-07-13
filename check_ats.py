import pandas as pd
from data.providers.finra import FinraProvider

finra = FinraProvider()
week = finra.get_latest_week()
data = finra.get_week_summary(week)
print('Columnas originales:', data.columns.tolist())

# Agrupar por símbolo y sumar volumen
ats_vol = data.groupby('issueSymbolIdentifier')['totalWeeklyShareQuantity'].sum().reset_index(name='ats_volume')
print('Columnas tras agrupar:', ats_vol.columns.tolist())
print('Muestra:')
print(ats_vol.head(5))

# Buscar XLF
xlf_row = ats_vol.loc[ats_vol['issueSymbolIdentifier'] == 'XLF']
print('\nXLF ATS volume:', xlf_row['ats_volume'].values[0] if len(xlf_row) > 0 else 'NO ENCONTRADO')
