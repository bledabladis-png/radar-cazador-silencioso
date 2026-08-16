"""
Proveedor Amundi para flujo primario y posiciones de ETFs UCITS.
Descarga metadatos (SHARES_OUT, NAV, AUM, fechas) y composición.
Calcula flujo primario estimado = ΔSHARES_OUT × NAV.
"""
import requests
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta

ISIN_LYXI = 'FR0010251744'
API_URL = 'https://www.amundietf.es/mapi/ProductAPI/getProductsData'
CACHE_DIR = Path('data/cache/amundi')
FUND_HISTORY_CSV = Path('outputs/history/amundi_lyxi_fund_history.csv')
HOLDINGS_HISTORY_CSV = Path('outputs/history/amundi_lyxi_holdings_history.csv')

HEADERS = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Origin': 'https://www.amundietf.es',
    'Referer': 'https://www.amundietf.es/',
    'User-Agent': 'Mozilla/5.0'
}

def normalize_amundi_date(value):
    """Convierte fechas Amundi (timestamp ms o ISO) a date."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value / 1000).date()
        except (ValueError, OSError):
            return None
    if isinstance(value, str):
        try:
            return pd.to_datetime(value).date()
        except:
            return None
    return None

def build_request_body(isin: str) -> dict:
    return {
        "context": {
            "countryCode": "ESP",
            "countryName": "Spain",
            "googleCountryCode": "ES",
            "domainName": "www.amundietf.es",
            "bcp47Code": "es-ES",
            "languageName": "Spanish",
            "languageCode": "es",
            "userProfileName": "RETAIL",
            "userProfileSlug": "retail"
        },
        "productIds": [isin],
        "characteristics": [
            "ISIN",
            "TICKER",
            "SHARE_MARKETING_NAME",
            "SHARES_OUT",
            "NAV",
            "AUM",
            "CURRENCY",
            "FUND_AUM",
            "POSITION_AS_OF_DATE",
            "NAV_DATE_DISPLAYED",
            "NAV_DATE_FOR_PERFORMANCE_CALCULATIONS",
            "NNA_DATA_DATE",
            "FUND_BREAKDOWNS_AS_OF_DATE"
        ],
        "historics": [],
        "metrics": [],
        "breakDown": {
            "aggregationFields": ["FUND_TOP10"]
        },
        "productType": "PRODUCT",
        "composition": {
            "compositionFields": [
                "date", "type", "bbg", "isin", "name", "weight",
                "quantity", "currency", "sector", "country", "countryOfRisk"
            ]
        }
    }

def download_fund_data(isin: str, force_download: bool = False) -> dict:
    """Descarga los datos de la API y los cachea."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f'{isin}.json'
    use_cache = (not force_download) and cache_file.exists()
    if use_cache:
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime > timedelta(hours=23):
            use_cache = False

    if use_cache:
        print(f'  Usando caché para {isin}')
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    print(f'  Descargando {isin} desde Amundi...')
    body = build_request_body(isin)
    try:
        r = requests.post(API_URL, json=body, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        products = data.get('products', [])
        if not products:
            raise ValueError('No products returned')
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(products[0], f, ensure_ascii=False, indent=2)
        print(f'  Guardado en caché: {cache_file}')
        return products[0]
    except Exception as e:
        print(f'  Error descargando {isin}: {e}')
        if cache_file.exists():
            print('  Usando caché existente pese al error.')
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

def save_daily_snapshot(product: dict, isin: str):
    """Guarda registros diarios de SHARES_OUT, NAV, AUM y composición."""
    if not product:
        print('  No hay datos de producto, no se guarda snapshot.')
        return

    chars = product.get('characteristics', {})
    # Fechas efectivas normalizadas
    shares_date = normalize_amundi_date(chars.get('POSITION_AS_OF_DATE'))
    nav_date = normalize_amundi_date(chars.get('NAV_DATE_DISPLAYED') or chars.get('NAV_DATE_FOR_PERFORMANCE_CALCULATIONS'))
    aum_date = normalize_amundi_date(chars.get('NNA_DATA_DATE') or chars.get('FUND_BREAKDOWNS_AS_OF_DATE'))

    shares_out = chars.get('SHARES_OUT')
    nav = chars.get('NAV')
    aum = chars.get('AUM')
    fund_name = chars.get('SHARE_MARKETING_NAME', 'Amundi ETF')
    currency = chars.get('CURRENCY', 'EUR')

    calculated_aum = shares_out * nav if (shares_out is not None and nav is not None) else None
    aum_error_pct = None
    if calculated_aum and aum:
        aum_error_pct = (calculated_aum - aum) / aum * 100

    # Fecha global: priorizamos shares_date, luego nav_date, luego aum_date
    as_of = shares_date or nav_date or aum_date
    if as_of is None:
        raise ValueError("Amundi no proporcionó fecha efectiva para el snapshot")

    fund_row = {
        'isin': isin,
        'fund_name': fund_name,
        'date': as_of,
        'shares_outstanding': shares_out,
        'nav': nav,
        'aum': aum,
        'currency': currency,
        'nav_date': nav_date,
        'shares_date': shares_date,
        'aum_date': aum_date,
        'calculated_aum': calculated_aum,
        'aum_error_pct': aum_error_pct
    }

    if FUND_HISTORY_CSV.exists():
        df_fund = pd.read_csv(FUND_HISTORY_CSV)
        df_fund = df_fund[df_fund['date'] != str(as_of)]
        df_fund = pd.concat([df_fund, pd.DataFrame([fund_row])], ignore_index=True)
    else:
        df_fund = pd.DataFrame([fund_row])
    df_fund.to_csv(FUND_HISTORY_CSV, index=False)
    print(f'  Fund history actualizado: {FUND_HISTORY_CSV}')
    if aum_error_pct is not None:
        print(f'  AUM CHECK: error {aum_error_pct:.6f}%')

    comp = product.get('composition', {})
    comp_data = comp.get('compositionData', [])
    if not comp_data:
        print('  No hay compositionData, no se guardan posiciones.')
        return

    rows = []
    for item in comp_data:
        c = item.get('compositionCharacteristics', item)
        rows.append({
            'date': c.get('date', as_of),
            'etf_isin': isin,
            'etf_name': fund_name,
            'holding_isin': c.get('isin'),
            'holding_bbg': c.get('bbg'),
            'holding_name': c.get('name'),
            'quantity': c.get('quantity'),
            'weight': c.get('weight'),
            'currency': c.get('currency'),
            'sector': c.get('sector'),
            'country': c.get('country'),
            'country_of_risk': c.get('countryOfRisk'),
        })

    if HOLDINGS_HISTORY_CSV.exists():
        df_holdings = pd.read_csv(HOLDINGS_HISTORY_CSV)
        df_holdings = df_holdings[df_holdings['date'] != str(as_of)]
        df_holdings = pd.concat([df_holdings, pd.DataFrame(rows)], ignore_index=True)
    else:
        df_holdings = pd.DataFrame(rows)
    df_holdings.to_csv(HOLDINGS_HISTORY_CSV, index=False)
    print(f'  Holdings history actualizado: {HOLDINGS_HISTORY_CSV} ({len(rows)} posiciones)')

def get_amundi_primary_flow(isin: str, force_download: bool = False) -> pd.DataFrame:
    """Descarga y guarda snapshot, devuelve últimos datos de flujo primario estimado."""
    product = download_fund_data(isin, force_download=force_download)
    save_daily_snapshot(product, isin)

    if FUND_HISTORY_CSV.exists():
        df = pd.read_csv(FUND_HISTORY_CSV, parse_dates=['date'])
        df = df.sort_values('date')
        if len(df) >= 2:
            df['shares_change'] = df['shares_outstanding'].diff()
            df['estimated_flow_eur'] = df['shares_change'] * df['nav']
            df['flow_pct_assets'] = df['estimated_flow_eur'] / df['aum']
            return df.tail(1)
        else:
            # Histórico insuficiente: devolver última fila con flujo NaN
            last = df.tail(1).copy()
            last['shares_change'] = pd.NA
            last['estimated_flow_eur'] = pd.NA
            last['flow_pct_assets'] = pd.NA
            return last
    return pd.DataFrame()

# Wrapper para LYXI
def get_amundi_lyxi_primary_flow(force_download: bool = False) -> pd.DataFrame:
    return get_amundi_primary_flow(ISIN_LYXI, force_download=force_download)

if __name__ == '__main__':
    df = get_amundi_lyxi_primary_flow(force_download=True)
    print('\nÚltima fila de flujo LYXI:')
    print(df)

