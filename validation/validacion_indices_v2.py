# -*- coding: utf-8 -*-
# Fase 0 v2 - Validacion de tickers desde CSV local
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import yfinance as yf
from datetime import datetime

holdings = pd.read_csv('data/index_holdings.csv')
print("Holdings cargados:", len(holdings))

INDICES = {
    'S&P 500':       {'index': '^GSPC', 'etf': 'SPY',  'expected': 20},
    'Dow Jones':     {'index': '^DJI',  'etf': 'DIA',  'expected': 10},
    'Nasdaq-100':    {'index': '^NDX',  'etf': 'QQQ',  'expected': 15},
    'Russell 2000':  {'index': '^RUT',  'etf': 'IWM',  'expected': 10},
    'Euro Stoxx 50': {'index': '^STOXX50E', 'etf': 'FEZ', 'expected': 10},
    'Ibex 35':       {'index': '^IBEX', 'etf': 'LYXI', 'expected': 10},
    'DAX 40':        {'index': '^GDAXI','etf': 'DAXEX','expected': 10},
    'FTSE 100':      {'index': '^FTSE', 'etf': 'ISF.L','expected': 10},
}

resultados = []

for nombre, datos in INDICES.items():
    print(f"Validando {nombre}...")
    idx_ticker = datos['index']
    etf_ticker = datos['etf']
    expected = datos['expected']

    # 1. Datos OHLCV del indice
    ohlcv_ok = False
    ohlcv_rows = 0
    try:
        idx_data = yf.download(idx_ticker, period='5y', progress=False, auto_adjust=True)
        ohlcv_rows = len(idx_data)
        ohlcv_ok = ohlcv_rows >= 200
    except:
        ohlcv_ok = False

    # 2. Tickers del CSV para este ETF
    tickers = holdings[holdings['etf'] == etf_ticker]['ticker'].tolist()[:expected]
    tickers_count = len(tickers)
    tickers_ok = tickers_count >= expected

    # 3. Descargabilidad de una muestra (primeros 3 tickers)
    descargables = 0
    muestra = tickers[:3]
    if muestra:
        try:
            test_data = yf.download(muestra, period='1mo', progress=False, auto_adjust=True)
            descargables = len(test_data.columns.unique(level=1)) if isinstance(test_data.columns, pd.MultiIndex) else len(muestra)
        except:
            pass

    resultados.append({
        'indice': nombre,
        'ohlcv_ok': ohlcv_ok,
        'tickers_ok': tickers_ok,
        'tickers_count': tickers_count,
        'descargables': descargables,
        'viable': ohlcv_ok and tickers_ok and descargables > 0
    })

# Informe
with open('outputs/validacion_indices_v2.md', 'w', encoding='utf-8') as f:
    f.write('# Validacion Fase 0 v2 - Modulo de Indices Internacionales\n\n')
    f.write(f'**Fecha:** {datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n')
    f.write('| Indice | OHLCV | Tickers (disp/esp) | Descargables | Viable |\n')
    f.write('|--------|-------|-------------------|--------------|--------|\n')
    for r in resultados:
        f.write(f"| {r['indice']} | {'OK' if r['ohlcv_ok'] else 'FALLO'} | {r['tickers_count']}/{INDICES[r['indice']]['expected']} | {r['descargables']} | {'SI' if r['viable'] else 'NO'} |\n")
    viables = [r for r in resultados if r['viable']]
    f.write(f'\n**Viables:** {len(viables)} de {len(resultados)}\n')
print("Informe generado: outputs/validacion_indices_v2.md")
