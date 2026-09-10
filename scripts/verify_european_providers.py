"""
Script temporal para verificar que GitHub Actions puede descargar
de Euronext y Xetra. Se elimina tras la verificacion.
"""
import os
import sys
import time
from pathlib import Path

# Asegurar imports desde raiz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_euronext(ticker):
    print('')
    print('=' * 60)
    print('TEST EURONEXT: ' + ticker)
    print('=' * 60)
    try:
        from data.providers.euronext_provider import EuronextProvider
        provider = EuronextProvider()
        if not provider.supports(ticker):
            print('[FAIL] ' + ticker + ' no soportado por EuronextProvider')
            return False
        t0 = time.time()
        df = provider.get_prices([ticker], nb_session=20, use_cache=False)
        elapsed = time.time() - t0
        if df is None or df.empty:
            print('[FAIL] Sin datos (tiempo: ' + str(round(elapsed, 1)) + 's)')
            return False
        rows = len(df)
        print('[OK] Descargadas ' + str(rows) + ' filas en ' + str(round(elapsed, 1)) + 's')
        print('     Columnas: ' + str(list(df.columns.get_level_values(0).unique())))
        print('     Primeras fechas: ' + str(df.index[:3].tolist()))
        print('     Ultimas fechas: ' + str(df.index[-3:].tolist()))
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('[FAIL] Excepcion: ' + str(e))
        return False


def test_xetra(ticker):
    print('')
    print('=' * 60)
    print('TEST XETRA: ' + ticker)
    print('=' * 60)
    try:
        from data.providers.xetra_provider import XetraProvider
        provider = XetraProvider()
        if not provider.supports(ticker):
            print('[FAIL] ' + ticker + ' no soportado por XetraProvider')
            return False
        t0 = time.time()
        df = provider.get_prices([ticker], use_cache=False)
        elapsed = time.time() - t0
        if df is None or df.empty:
            print('[FAIL] Sin datos (tiempo: ' + str(round(elapsed, 1)) + 's)')
            return False
        rows = len(df)
        print('[OK] Descargadas ' + str(rows) + ' filas en ' + str(round(elapsed, 1)) + 's')
        print('     Columnas: ' + str(list(df.columns.get_level_values(0).unique())))
        print('     Primeras fechas: ' + str(df.index[:3].tolist()))
        print('     Ultimas fechas: ' + str(df.index[-3:].tolist()))
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('[FAIL] Excepcion: ' + str(e))
        return False


def main():
    ticker_euronext = os.environ.get('TICKER_EURONEXT', 'AI.PA')
    ticker_xetra = os.environ.get('TICKER_XETRA', 'SAP.DE')

    print('Entorno: GitHub Actions = ' + str(os.environ.get('GITHUB_ACTIONS', 'no')))
    print('Python: ' + sys.version)
    print('Ticker Euronext: ' + ticker_euronext)
    print('Ticker Xetra: ' + ticker_xetra)

    ok_euronext = test_euronext(ticker_euronext)
    ok_xetra = test_xetra(ticker_xetra)

    print('')
    print('=' * 60)
    print('RESUMEN')
    print('=' * 60)
    print('Euronext: ' + ('[OK]' if ok_euronext else '[FAIL]'))
    print('Xetra:    ' + ('[OK]' if ok_xetra else '[FAIL]'))

    if ok_euronext and ok_xetra:
        print('')
        print('[OK] GitHub Actions puede descargar de ambas fuentes.')
        print('[OK] Podemos proceder con Europa primero.')
        sys.exit(0)
    else:
        print('')
        print('[FAIL] Algun proveedor no funciona en GitHub Actions.')
        print('[WARN] Revisar antes de cambiar arquitectura.')
        sys.exit(1)


if __name__ == '__main__':
    main()
