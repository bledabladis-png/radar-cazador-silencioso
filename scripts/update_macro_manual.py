"""
update_macro_manual.py -- Regenera los CSVs de data/macro_manual/ desde FRED.

Fuente: FRED (Federal Reserve Economic Data), acceso publico via
pandas_datareader.

Ejecucion:
- Workflow diario update_macro_manual.yml (06:00 UTC)
- Manual: py scripts/update_macro_manual.py

Notas:
- Leading_Index de actividad.csv NO se descarga: USSLIND discontinuada
  en FRED desde 2020-02.
- Cada CSV preserva las columnas y el orden originales.
"""
from pathlib import Path
import sys
import time

import pandas as pd
import pandas_datareader.data as web


START_DATE = '2000-01-01'
OUTPUT_DIR = Path('data/macro_manual')

# Mapeo CSV -> lista ordenada de (nombre_columna_CSV, serie_FRED)
MACRO_MANUAL_MAP = {
    '10y3m.csv': [
        ('T10Y3M', 'T10Y3M'),
    ],
    'actividad.csv': [
        ('Industrial_Production_Total', 'INDPRO'),
        ('Industrial_Production_Manufacturing', 'IPMAN'),
        ('Retail_Sales', 'RSAFS'),
        # Leading_Index eliminado: USSLIND discontinuada 2020-02
    ],
    'commercial_paper.csv': [
        ('COMPOUT', 'COMPOUT'),
    ],
    'credit_oas.csv': [
        ('CreditOAS', 'BAMLC0A0CM'),
    ],
    'discount_rate.csv': [
        ('DPRIME', 'DPRIME'),
    ],
    'empleo.csv': [
        ('NonFarm_Payrolls', 'PAYEMS'),
        ('Unemployment_Rate', 'UNRATE'),
        ('Initial_Claims', 'ICSA'),
        ('Continuing_Claims', 'CCSA'),
        ('Avg_Hourly_Earnings', 'CES0500000003'),
        ('Total_Private_Employees', 'USPRIV'),
        ('Manufacturing_Employees', 'MANEMP'),
    ],
    'inflacion.csv': [
        ('CPI', 'CPIAUCSL'),
        ('Core_CPI', 'CPILFESL'),
        ('PCE', 'PCEPI'),
        ('Core_PCE', 'PCEPILFE'),
        ('Inf_Expect_5Y', 'T5YIFR'),
        ('Breakeven_10Y', 'T10YIE'),
    ],
    'iorb.csv': [
        ('IORB', 'IORB'),
    ],
    'nfci.csv': [
        ('NFCI', 'NFCI'),
    ],
    'rrpp.csv': [
        ('RRPONTSYD', 'RRPONTSYD'),
    ],
    'sofr.csv': [
        ('SOFR', 'SOFR'),
    ],
    'walcl.csv': [
        ('WALCL', 'WALCL'),
    ],
}


def download_series(series_id, retries=3):
    """Descarga una serie FRED. Devuelve pd.Series o None."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            df = web.DataReader(series_id, 'fred', start=START_DATE)
            if df is None or df.empty:
                return None
            s = df.iloc[:, 0]
            s.name = None  # El nombre se asigna despues
            return s
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(2 * attempt)
    print('  [FAIL] ' + series_id + ': ' + str(last_exc))
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_ok = 0
    total_fail = 0
    total_csv = 0

    for csv_name, series_list in MACRO_MANUAL_MAP.items():
        print('=== ' + csv_name + ' ===')
        data = {}
        for col_name, fred_id in series_list:
            print('  Descargando ' + col_name + ' <- ' + fred_id)
            s = download_series(fred_id)
            if s is None:
                total_fail += 1
                continue
            data[col_name] = s
            total_ok += 1

        if not data:
            print('  [WARN] Sin datos, omitiendo ' + csv_name)
            continue

        df = pd.DataFrame(data)
        df.index.name = 'date'
        df = df.reset_index()
        # Normalizar formato de fecha
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        out_path = OUTPUT_DIR / csv_name
        df.to_csv(out_path, index=False)
        print('  Guardado: ' + str(out_path)
              + ' (' + str(len(df)) + ' filas, '
              + str(len(df.columns) - 1) + ' series)')
        total_csv += 1

    print()
    print('=' * 60)
    print('RESUMEN')
    print('=' * 60)
    print('Series OK:    ' + str(total_ok))
    print('Series FAIL:  ' + str(total_fail))
    print('CSVs escritos: ' + str(total_csv) + '/' + str(len(MACRO_MANUAL_MAP)))

    return 0 if total_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
