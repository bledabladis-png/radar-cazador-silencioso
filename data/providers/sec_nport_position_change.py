"""
Calcula cambios de posiciones N-PORT entre reportes consecutivos por fondo y activo.
"""
import pandas as pd
from pathlib import Path

POSITIONS_PATH = Path('outputs/history/sec_nport_positions.csv')
OUTPUT_PATH = Path('outputs/history/sec_nport_position_change.csv')

def main():
    df = pd.read_csv(POSITIONS_PATH, parse_dates=['REPORT_DATE', 'FILING_DATE'])
    print(f'Posiciones leídas: {len(df)}')

    # Clave de seguridad: ISIN si existe, si no CUSIP
    df['SECURITY_KEY'] = df['IDENTIFIER_ISIN'].fillna(df['ISSUER_CUSIP'])

    # Ordenar y calcular cambio por fondo y seguridad
    df = df.sort_values(['CIK','SECURITY_KEY','REPORT_DATE'])
    df['PREV_BALANCE'] = df.groupby(['CIK','SECURITY_KEY'])['BALANCE'].shift(1)
    df['POSITION_CHANGE'] = df['BALANCE'] - df['PREV_BALANCE']
    df['POSITION_CHANGE_PCT'] = df['POSITION_CHANGE'] / df['PREV_BALANCE'].replace(0, pd.NA) * 100

    # Conservar solo filas con cambio disponible
    change = df.dropna(subset=['POSITION_CHANGE']).copy()

    # Seleccionar columnas finales
    cols = [
        'REPORT_DATE','FILING_DATE','CIK','REGISTRANT_NAME','HOLDING_ID',
        'SECURITY_KEY','ISSUER_NAME','BALANCE','PREV_BALANCE',
        'POSITION_CHANGE','POSITION_CHANGE_PCT','CURRENCY_VALUE','PERCENTAGE'
    ]
    change = change[cols]

    change.to_csv(OUTPUT_PATH, index=False)
    print(f'Guardado en {OUTPUT_PATH}')
    print(f'Cambios calculados: {len(change)}')
    print(change.head(20).to_string(index=False))

if __name__ == '__main__':
    main()
