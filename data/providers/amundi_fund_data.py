"""
Proveedor Amundi para flujo primario de LYXI.
Descarga series históricas de SHARES_OUT, NAV y AUM desde la API oficial.
Calcula ETF Primary Flow = ΔSharesOutstanding × NAV.
"""
import requests
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from ._fund_flow_utils import fund_flow_robust_zscore_with_regime
from src.utils import append_dedup

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
    # Fix N (2026-10-01): descarga SIEMPRE. Cache solo como fallback
    # si la descarga falla. Antes: `mtime < 23h -> cache-hit`, que
    # perdia el dato si la fuente publicaba despues del ultimo run.
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
        # A5-79 (2026-09-28): escritura atomica tmp + replace.
        _tmp = cache_file.with_suffix(cache_file.suffix + '.tmp')
        with open(_tmp, 'w', encoding='utf-8') as f:
            json.dump(product, f, ensure_ascii=False, indent=2)
        _tmp.replace(cache_file)
        print(f'  Guardado en caché: {cache_file}')
        return product
    except (requests.RequestException, ValueError, KeyError,
            TypeError, OSError, json.JSONDecodeError) as e:
        print(f'  Error descargando histórico {isin}: {e}')
        if cache_file.exists():
            # A5-72 (2026-09-28): WARN explicito con mtime al caer a
            # cache obsoleta.
            _mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            _age_h = (datetime.now() - _mtime).total_seconds() / 3600.0
            print(f'  [WARN] {isin}: usando cache OBSOLETA pese al error '
                  f'(mtime={_mtime.isoformat(timespec="minutes")}, '
                  f'{_age_h:.1f}h). El dato puede estar desactualizado.')
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

    # Normalización robusta (mediana/MAD, clip ±5).
    # Contrato común con DAXEX, ISF e IWM (auditor 2026-09-27).
    df['flow_zscore'], df['flow_zscore_regime'] = fund_flow_robust_zscore_with_regime(
        df['flow_pct_assets'], window=120, min_periods=20,
    )
    df['flow_5d'] = df['estimated_flow_eur'].rolling(5).mean()
    df['flow_20d'] = df['estimated_flow_eur'].rolling(20).mean()

    return df

def get_amundi_lyxi_primary_flow(force_download: bool = False) -> pd.DataFrame:
    """Descarga histórico, calcula flujo y devuelve la última fila."""
    # Rango amplio para intentar obtener máximo histórico
    start_date = '2018-01-01'
    # A5-73 (2026-09-28): WONT FIX razonado. now() define la cota
    # superior del rango de descarga. La fecha de observacion real
    # viene del dataset. Propagar reference_date requiriria cambiar
    # get_amundi_lyxi_primary_flow + su caller en flows_primary.py.
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
        'flow_zscore', 'flow_zscore_regime', 'flow_5d', 'flow_20d'
    ]
    # Fix D (2026-09-30): preservar historico con append_dedup.
    # Mismo bug que ssga_fund_data: sobrescribia sin comparar con
    # el existente. Si el proveedor devolvia menos filas, se perdian.
    _df_to_write = df[cols]
    if HISTORY_CSV.exists():
        try:
            _hist_existing = pd.read_csv(HISTORY_CSV)
            _df_to_write = append_dedup(
                _hist_existing, _df_to_write, ["date"], sort_by=["date"]
            )
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as _e:
            print(f'  [WARN] amundi existente ilegible: {_e}')
    _tmp_hist = HISTORY_CSV.with_suffix(HISTORY_CSV.suffix + '.tmp')
    _df_to_write.to_csv(_tmp_hist, index=False, lineterminator='\n')
    _tmp_hist.replace(HISTORY_CSV)
    print(f'  Histórico guardado en {HISTORY_CSV}')
    print(f'  Total filas: {len(df)}')

    if not df.empty:
        print(df.tail(5).to_string(index=False))

    return df.tail(1)

if __name__ == '__main__':
    df = get_amundi_lyxi_primary_flow(force_download=True)
    print('\nÚltima fila de flujo LYXI:')
    print(df)

