"""
Proveedor Amundi para flujo primario de LYXI.
Descarga series históricas de SHARES_OUT, NAV y AUM desde la API oficial.
Calcula ETF Primary Flow = ΔSharesOutstanding × NAV.
"""
import requests
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta

ISIN_LYXI = 'FR0010251744'
API_URL = 'https://www.amundietf.es/mapi/ProductAPI/getProductsData'
CACHE_DIR = Path('data/cache/amundi')
HISTORY_CSV = Path('outputs/history/amundi_lyxi_primary_flow.csv')

HEADERS = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Origin': 'https://www.amundietf.es',
    'Referer': 'https://www.amundietf.es/',
    'User-Agent': 'Mozilla/5.0'
}

def build_historical_request(isin: str, start_date: str, end_date: str) -> dict:
    """Construye el body con las tres series históricas."""
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
            "SHARE_MARKETING_NAME",
            "SHARES_OUT",
            "NAV",
            "AUM",
            "CURRENCY"
        ],
        "historics": [
            {
                "indicator": "sharesOut",
                "startDate": f"{start_date}T00:00:00.000Z",
                "endDate": f"{end_date}T23:59:59.000Z"
            },
            {
                "indicator": "officialNav",
                "startDate": f"{start_date}T00:00:00.000Z",
                "endDate": f"{end_date}T23:59:59.000Z"
            },
            {
                "indicator": "fundAumInMCcy",
                "startDate": f"{start_date}T00:00:00.000Z",
                "endDate": f"{end_date}T23:59:59.000Z"
            }
        ],
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

def download_historical_data(isin: str, start_date: str, end_date: str) -> dict:
    """Descarga datos históricos y los cachea por fecha de consulta."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f'{isin}_hist_{start_date}_{end_date}.json'
    use_cache = cache_file.exists()
    if use_cache:
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime > timedelta(hours=23):
            use_cache = False

    if use_cache:
        print(f'  Usando caché histórico para {isin}')
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    print(f'  Descargando histórico {isin} desde Amundi...')
    body = build_historical_request(isin, start_date, end_date)
    try:
        r = requests.post(API_URL, json=body, headers=HEADERS, timeout=60)
        r.raise_for_status()
        data = r.json()
        products = data.get('products', [])
        if not products:
            raise ValueError('No products returned')
        product = products[0]
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(product, f, ensure_ascii=False, indent=2)
        print(f'  Guardado en caché: {cache_file}')
        return product
    except Exception as e:
        print(f'  Error descargando histórico {isin}: {e}')
        if cache_file.exists():
            print('  Usando caché existente pese al error.')
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

def parse_historical_series(product: dict) -> pd.DataFrame:
    """Extrae y une las tres series por fecha (timestamp ms -> date)."""
    if not product:
        return pd.DataFrame()

    historics = product.get('historics', [])
    series_dict = {}

    for hist in historics:
        indicator = hist.get('indicator')
        data = hist.get('historicalData') or []
        if not data:
            continue
        df_series = pd.DataFrame(data)
        # La API devuelve 'date' (timestamp ms) y 'data' (valor)
        df_series['date'] = pd.to_datetime(df_series['date'], unit='ms').dt.date
        df_series = df_series.rename(columns={'data': indicator})
        # Conservar solo date y el indicador
        df_series = df_series[['date', indicator]]
        series_dict[indicator] = df_series

    required = ['sharesOut', 'officialNav']
    missing = [k for k in required if k not in series_dict]
    if missing:
        raise ValueError(f'Faltan series históricas: {missing}')

    # Unir por fecha
    df = series_dict['sharesOut'].merge(
        series_dict['officialNav'],
        on='date',
        how='outer'
    )
    if 'fundAumInMCcy' in series_dict:
        df = df.merge(series_dict['fundAumInMCcy'], on='date', how='left')

    df = df.sort_values('date').reset_index(drop=True)
    return df

def compute_primary_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula flujo primario y métricas derivadas."""
    if df.empty:
        return df

    df = df.copy()
    df['shares_outstanding'] = pd.to_numeric(df['sharesOut'], errors='coerce')
    df['nav'] = pd.to_numeric(df['officialNav'], errors='coerce')
    df['fund_aum'] = pd.to_numeric(df['fundAumInMCcy'], errors='coerce')
    df['class_aum'] = df['shares_outstanding'] * df['nav']  # AUM de la clase

    # Flujo primario
    df['shares_change'] = df['shares_outstanding'].diff()
    df['estimated_flow_eur'] = df['shares_change'] * df['nav']
    df['flow_pct_assets'] = df['estimated_flow_eur'] / df['class_aum']  # decimal

    # Normalización robusta (media y desviación estándar, clip ±3)
    mean = df['flow_pct_assets'].rolling(120, min_periods=20).mean()
    std = df['flow_pct_assets'].rolling(120, min_periods=20).std()
    df['flow_zscore'] = ((df['flow_pct_assets'] - mean) / (std + 1e-9)).clip(-3, 3)
    df['flow_5d'] = df['estimated_flow_eur'].rolling(5).mean()
    df['flow_20d'] = df['estimated_flow_eur'].rolling(20).mean()

    return df

def get_amundi_lyxi_primary_flow(force_download: bool = False) -> pd.DataFrame:
    """Descarga histórico, calcula flujo y devuelve la última fila."""
    # Rango amplio para intentar obtener máximo histórico
    start_date = '2018-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    product = download_historical_data(ISIN_LYXI, start_date, end_date)
    if not product:
        return pd.DataFrame()

    print('  Procesando series históricas...')
    df = parse_historical_series(product)
    if df.empty:
        print('  No se obtuvieron series históricas.')
        return pd.DataFrame()

    df = compute_primary_flow(df)

    # Guardar CSV completo
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        'date', 'shares_outstanding', 'nav', 'fund_aum', 'class_aum',
        'shares_change', 'estimated_flow_eur', 'flow_pct_assets',
        'flow_zscore', 'flow_5d', 'flow_20d'
    ]
    df[cols].to_csv(HISTORY_CSV, index=False)
    print(f'  Histórico guardado en {HISTORY_CSV}')
    print(f'  Total filas: {len(df)}')

    if not df.empty:
        print(df.tail(5).to_string(index=False))

    return df.tail(1)

if __name__ == '__main__':
    df = get_amundi_lyxi_primary_flow(force_download=True)
    print('\nÚltima fila de flujo LYXI:')
    print(df)

