# -*- coding: utf-8 -*-
# Fase 0 - Validacion de fuentes para modulo de indices internacionales
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import yfinance as yf
from datetime import datetime

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

    # 1. Datos OHLCV del indice (5y)
    ohlcv_ok = False
    ohlcv_rows = 0
    try:
        idx_data = yf.download(idx_ticker, period='5y', progress=False, auto_adjust=True)
        ohlcv_rows = len(idx_data)
        ohlcv_ok = ohlcv_rows >= 200  # minimo para Wyckoff
    except Exception as e:
        ohlcv_ok = False
        ohlcv_error = str(e)

    # 2. Holdings del ETF proxy
    holdings_ok = False
    holdings_count = 0
    holdings_tickers = []
    try:
        etf = yf.Ticker(etf_ticker)
        holdings = etf.get_etf_holdings()
        if holdings is not None and not holdings.empty:
            holdings_count = len(holdings)
            holdings_ok = holdings_count >= expected
            # Guardar tickers (asumiendo columna 'Symbol' o 'Ticker')
            col = 'Symbol' if 'Symbol' in holdings.columns else holdings.columns[0]
            holdings_tickers = holdings[col].dropna().tolist()[:expected]
    except Exception as e:
        holdings_ok = False
        holdings_error = str(e)

    # 3. Duplicados e invalidos
    duplicados = len(holdings_tickers) != len(set(holdings_tickers)) if holdings_tickers else False
    tickers_invalidos = [t for t in holdings_tickers if not isinstance(t, str) or len(t) > 10] if holdings_tickers else []

    resultados.append({
        'indice': nombre,
        'index_ticker': idx_ticker,
        'etf_ticker': etf_ticker,
        'ohlcv_ok': ohlcv_ok,
        'ohlcv_rows': ohlcv_rows,
        'holdings_ok': holdings_ok,
        'holdings_count': holdings_count,
        'expected': expected,
        'duplicados': duplicados,
        'invalidos': len(tickers_invalidos),
        'tickers_muestra': ', '.join(holdings_tickers[:5]) if holdings_tickers else 'N/A',
        'viable': ohlcv_ok and holdings_ok
    })

# Generar informe Markdown
with open('outputs/audit/validacion_indices.md', 'w', encoding='utf-8') as f:
    f.write('# Validacion de fuentes - Modulo de Indices Internacionales\n\n')
    f.write(f'**Fecha:** {datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n')
    f.write('| Indice | Ticker | ETF | OHLCV (filas) | Holdings (disp/esp) | Duplicados | Inválidos | Viable |\n')
    f.write('|--------|--------|-----|---------------|---------------------|------------|-----------|--------|\n')
    for r in resultados:
        ohlcv_str = f'OK ({r["ohlcv_rows"]})' if r['ohlcv_ok'] else 'FALLO'
        hold_str = f'OK ({r["holdings_count"]}/{r["expected"]})' if r['holdings_ok'] else f'FALLO ({r["holdings_count"]}/{r["expected"]})'
        viable = 'SI' if r['viable'] else 'NO'
        f.write(f"| {r['indice']} | {r['index_ticker']} | {r['etf_ticker']} | {ohlcv_str} | {hold_str} | {r['duplicados']} | {r['invalidos']} | {viable} |\n")

    f.write('\n## Muestra de tickers por indice\n\n')
    for r in resultados:
        f.write(f"- **{r['indice']}:** {r['tickers_muestra']}\n")

    f.write('\n## Conclusion\n\n')
    viables = [r for r in resultados if r['viable']]
    no_viables = [r for r in resultados if not r['viable']]
    f.write(f'- Indices viables: {len(viables)} ({", ".join([r["indice"] for r in viables]) or "ninguno"})\n')
    f.write(f'- Indices NO viables: {len(no_viables)} ({", ".join([r["indice"] for r in no_viables]) or "ninguno"})\n')
    if no_viables:
        f.write('\n**Acciones requeridas:** Revisar manualmente los indices no viables antes de continuar con la Fase 1.\n')

print("Informe generado: outputs/audit/validacion_indices.md")
print(f"Viables: {len(viables)}, No viables: {len(no_viables)}")
