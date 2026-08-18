# -*- coding: utf-8 -*-
"""
Proveedor Invesco QQQ — snapshot y NAV histórico.
Usa curl_cffi para evitar bloqueos 406.
No calcula flujo primario por falta de histórico de shares.
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import tempfile, os, sys, argparse
import time

TICKER = "QQQ"
CUSIP = "46090E103"
BASE_URL = "https://dng-api.invesco.com/cache/v1/accounts/en_US/shareclasses/46090E103"
URL_NAVS = BASE_URL + "/navs?idType=cusip&productType=ETF"
URL_PRICES = BASE_URL + "/prices?idType=cusip&variationType=priceListing&productType=ETF&productSubType=ETF"
URL_KEY_STATS = BASE_URL + "/keyStats?idType=cusip&productType=ETF"
URL_PERF = BASE_URL + "/performance/standard?idType=cusip&productType=ETF&performanceSubType=annualized&performancePeriod=monthly"

CACHE_DIR = Path("data/cache")
HISTORY_DIR = Path("outputs/history")
PERF_CSV = HISTORY_DIR / "invesco_qqq_performance.csv"
NAV_CACHE = CACHE_DIR / "qqq_navs.json"
PRICES_CACHE = CACHE_DIR / "qqq_prices.json"
KEY_STATS_CACHE = CACHE_DIR / "qqq_keystats.json"
NAV_HISTORY = HISTORY_DIR / "invesco_qqq_nav_historical.csv"
NAV_BUSINESS = HISTORY_DIR / "invesco_qqq_nav_business_days.csv"
SNAPSHOT = HISTORY_DIR / "invesco_qqq_snapshot.csv"

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "es-ES,es;q=0.9",
    "cache-control": "no-cache",
    "origin": "https://www.invesco.com",
    "pragma": "no-cache",
    "referer": "https://www.invesco.com/",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    ),
}

def _download_json(url, retries=3, backoff=2):
    """Download JSON using system curl with retry/backoff."""
    import subprocess
    import shutil

    headers = [
        "Accept: application/json, text/plain, */*",
        "Accept-Language: en-US,en;q=0.9",
        "Cache-Control: no-cache",
        "Origin: https://www.invesco.com",
        "Pragma: no-cache",
        "Referer: https://www.invesco.com/",
        "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
        f"resourcepath: {url}",
        "appid: invesco",
        "componenttype: ETF",
    ]

    curl = shutil.which("curl") or "curl"

    for attempt in range(retries):
        command = [curl, "-sS", "--fail-with-body", url]
        for header in headers:
            command.extend(["-H", header])

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )

        if result.returncode == 0:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"JSON invalid: {exc}") from exc

        if attempt < retries - 1:
            time.sleep(backoff * attempt)

    raise RuntimeError(
        result.stderr.strip() or result.stdout.strip() or "curl failed"
    )


def load_or_cache(url, cache_file, force=False, max_age_hours=23):
    """Read cache or download with TTL."""
    if cache_file.exists() and not force:
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age = datetime.now() - mtime
        if age <= timedelta(hours=max_age_hours):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        print(f"Cache stale ({age}). Downloading...")

    data = _download_json(url)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data

def extract_nav(payload):
    line_chart = payload.get('lineChartData', [])
    nav_series = None
    for item in line_chart:
        if item.get('type','').upper() == 'NAV':
            nav_series = item
            break
    if not nav_series:
        raise RuntimeError('No se encontró serie NAV')
    rows = nav_series.get('data', [])
    records = []
    for row in rows:
        date = pd.to_datetime(row.get('date'), errors='coerce')
        nav = pd.to_numeric(row.get('value'), errors='coerce')
        if pd.isna(date) or pd.isna(nav):
            continue
        records.append({'date': date.date(), 'nav': float(nav)})
    df = pd.DataFrame(records)
    df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
    return df

def parse_key_stats(payload):
    result = {}
    for item in payload.get('keyStats', []):
        name = item.get('name')
        if name:
            result[name] = {'value': item.get('value'), 'as_of_date': item.get('asOfDate')}
    return result

def download_performance(force=False):
    """Descarga y guarda performance anualizada oficial."""
    cache_file = CACHE_DIR / "qqq_performance.json"
    payload = load_or_cache(URL_PERF, cache_file, force)
    entries = payload.get('annualizedPerformance', [])
    df = pd.DataFrame(entries)
    df['effectiveDate'] = payload.get('effectiveDate')
    df['performancePeriod'] = 'monthly'
    df.to_csv(PERF_CSV, index=False)
    print(f'Performance guardada: {PERF_CSV}')
    return df

def main(force=False):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print('Descargando NAV histórico...')
    nav_payload = load_or_cache(URL_NAVS, NAV_CACHE, force)
    time.sleep(1)
    nav_df = extract_nav(nav_payload)
    nav_df.to_csv(NAV_HISTORY, index=False)
    business = nav_df[pd.to_datetime(nav_df['date']).dt.dayofweek < 5].reset_index(drop=True)
    business.to_csv(NAV_BUSINESS, index=False)

    print('Descargando snapshot...')
    prices_payload = load_or_cache(URL_PRICES, PRICES_CACHE, force)
    time.sleep(1)
    key_payload = load_or_cache(URL_KEY_STATS, KEY_STATS_CACHE, force)
    time.sleep(1)
    stats = parse_key_stats(key_payload)

    snapshot = {
        'ticker': TICKER,
        'cusip': CUSIP,
        'effective_date': prices_payload.get('effectiveDate'),
        'nav': prices_payload.get('nav'),
        'market_value': prices_payload.get('marketValue'),
        'shares_outstanding': prices_payload.get('sharesOutstanding'),
        'volume_30d_avg': prices_payload.get('30dayAverageTradingVolume'),
        'volume_prev_day': prices_payload.get('previousDayTradingVolume'),
        'open': prices_payload.get('openingPrice'),
        'close': prices_payload.get('closingPrice'),
        'bid_ask_midpoint': prices_payload.get('bidAskMidpoint'),
        'premium_discount_pct': prices_payload.get('bidAskMidpointPremiumDiscountPercentage'),
        'ytd': stats.get('ytd', {}).get('value'),
        'sec_yield_30d': stats.get('secYield30Day', {}).get('value'),
    }
    pd.DataFrame([snapshot]).to_csv(SNAPSHOT, index=False)

    # Descargar performance
    try:
        download_performance(force)
        time.sleep(1)
    except Exception as e:
        print(f'Performance QQQ no descargada: {e}')

    print('Archivos QQQ generados:')
    print(f'  {NAV_HISTORY} ({len(nav_df)} filas)')
    print(f'  {NAV_BUSINESS} ({len(business)} filas)')
    print(f'  {SNAPSHOT} (1 fila)')
    print('Snapshot QQQ:')
    print(snapshot)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Forzar descarga')
    args = parser.parse_args()
    main(force=args.force)
