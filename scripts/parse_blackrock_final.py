import requests
import csv
import yfinance as yf
import time

urls = {
    'DAXEX': 'https://www.blackrock.com/es/profesionales/productos/251464/ishares-dax-ucits-etf-de-fund/1497267045693.ajax?fileType=csv&fileName=DAXEX_holdings&dataType=fund',
    'ISF.L': 'https://www.blackrock.com/es/profesionales/productos/251795/ishares-ftse-100-ucits-etf-inc-fund/1497267045693.ajax?fileType=csv&fileName=ISF_holdings&dataType=fund',
}

SPECIAL_MAP = {
    'AIR': 'AIR.PA',
    'BT.A': 'BT.L',
}

NON_EQUITY_KEYWORDS = [
    'CASH', 'FUTURE', 'FX', 'CURRENCY', 'FUT', 'FORWARD', 'WARRANT', 'OPTION'
]

def clean_text(v):
    return (v or '').strip()

def parse_blackrock(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    texto = r.content.decode('utf-8-sig')
    lineas = texto.splitlines()
    header_idx = None
    for i, linea in enumerate(lineas):
        if linea.strip().startswith('Ticker'):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError('No se encontró cabecera Ticker')
    lector = csv.DictReader(lineas[header_idx:], delimiter=',')
    return list(lector)

def get_suffix(exchange):
    exchange = clean_text(exchange).lower()
    if 'xetra' in exchange or 'boerse berlin' in exchange or 'frankfurt' in exchange:
        return '.DE'
    if 'london' in exchange:
        return '.L'
    if 'euronext' in exchange or 'amsterdam' in exchange:
        return '.AS'
    if 'madrid' in exchange:
        return '.MC'
    if 'milan' in exchange:
        return '.MI'
    return ''

def map_yahoo(ticker, exchange):
    t = clean_text(ticker)
    if t in SPECIAL_MAP:
        return SPECIAL_MAP[t]
    if t.endswith('.'):
        t = t[:-1]
    suff = get_suffix(exchange)
    if suff and not t.endswith(suff):
        return t + suff
    return t

def validate_yahoo(ticker):
    try:
        y = yf.Ticker(ticker)
        hist = y.history(period='5d', auto_adjust=False)
        return not hist.empty
    except:
        return False

for etf, url in urls.items():
    print(f'\n===== {etf} =====')
    filas = parse_blackrock(url)
    holdings = []

    for f in filas:
        ticker = clean_text(f.get('Ticker'))
        name = clean_text(f.get('Name'))
        weight_str = clean_text(f.get('Weight (%)'))
        asset_class = clean_text(f.get('Asset Class'))
        type_field = clean_text(f.get('Type'))

        # Filtro: solo Equity
        if asset_class.lower() != 'equity' and type_field.upper() != 'EQUITY':
            continue

        # Filtrar no equity por keywords en name/ticker
        combined = f'{ticker} {name}'.upper()
        if any(k in combined for k in NON_EQUITY_KEYWORDS):
            continue

        try:
            weight = float(weight_str.replace(',', '.'))
        except:
            continue

        yahoo_ticker = map_yahoo(ticker, f.get('Exchange'))

        if ticker in SPECIAL_MAP and yahoo_ticker != SPECIAL_MAP[ticker]:
            yahoo_ticker = SPECIAL_MAP[ticker]

        valid = validate_yahoo(yahoo_ticker)
        if valid:
            holdings.append({
                'etf': etf,
                'ticker': yahoo_ticker,
                'name': name,
                'weight': weight,
            })
            print(f'OK: {ticker:6s} -> {yahoo_ticker}')
        else:
            print(f'FAIL: {ticker:6s} -> {yahoo_ticker} [excluido]')
        time.sleep(0.1)

    # Ordenar por peso descendente y tomar top 10
    holdings.sort(key=lambda x: x['weight'], reverse=True)
    top10 = holdings[:10]

    out = f'outputs/holdings/{etf}_final_holdings.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        escritor = csv.writer(f)
        escritor.writerow(['etf','ticker','name','weight'])
        for h in top10:
            escritor.writerow([h['etf'], h['ticker'], h['name'], f'{h["weight"]:.6f}'])
    print(f'Guardado: {out} ({len(holdings)} validos, top10 exportado)')
