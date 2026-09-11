"""
verify_fred_access.py -- Test de acceso a FRED desde GitHub Actions.
TEMPORAL: eliminar tras verificar.
"""
import os
import sys
import time
import traceback

import pandas_datareader.data as web


SERIES = [
    ('WALCL',      'Fed Balance Sheet (weekly)'),
    ('SOFR',       'SOFR rate (daily)'),
    ('CPIAUCSL',   'CPI (monthly)'),
    ('T5YIFR',     '5Y5Y Forward inflation (daily)'),
    ('BAMLC0A0CM', 'IG OAS (daily)'),
]


def main():
    print('Entorno: GitHub Actions =', os.environ.get('GITHUB_ACTIONS', 'no'))
    print('Python:', sys.version)
    print()

    results = []
    for series_id, desc in SERIES:
        print('=' * 60)
        print(f'{series_id} - {desc}')
        print('=' * 60)
        t0 = time.time()
        try:
            df = web.DataReader(series_id, 'fred', start='2024-01-01')
            elapsed = time.time() - t0
            if df.empty:
                print(f'[FAIL] Vacio ({elapsed:.1f}s)')
                results.append((series_id, False, 0, None))
                continue
            rows = len(df)
            last_date = df.index[-1].date()
            last_val = float(df.iloc[-1, 0])
            print(f'[OK] {rows} filas en {elapsed:.1f}s')
            print(f'     Ultima: {last_date} = {last_val}')
            results.append((series_id, True, rows, last_date))
        except Exception as e:
            print(f'[FAIL] {str(e)[:120]}')
            traceback.print_exc()
            results.append((series_id, False, 0, None))
        print()

    print('=' * 60)
    print('RESUMEN')
    print('=' * 60)
    ok = sum(1 for r in results if r[1])
    print(f'OK: {ok}/{len(results)}')
    for series_id, success, rows, last_date in results:
        status = '[OK]' if success else '[FAIL]'
        if success:
            print(f'  {status} {series_id}: {rows} filas, ultima {last_date}')
        else:
            print(f'  {status} {series_id}')

    if ok == len(results):
        print()
        print('[OK] GitHub Actions puede acceder a FRED sin problemas.')
        return 0
    else:
        print()
        print('[FAIL] Algunas series fallaron.')
        return 1


if __name__ == '__main__':
    sys.exit(main())
