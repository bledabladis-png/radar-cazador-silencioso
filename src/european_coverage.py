"""
european_coverage.py -- Reporte de cobertura de tickers europeos
(Euronext + Xetra + BME).
Genera:
  - outputs/audit/european_coverage.md (informe legible)
  - outputs/history/european_coverage.csv (historico append)
"""
from pathlib import Path

import pandas as pd

from data.providers.euronext_provider import EuronextProvider
from data.providers.xetra_provider import XetraProvider
from data.providers.bme_provider import BMEProvider

OUTPUT_MD = Path('outputs/audit/european_coverage.md')
OUTPUT_CSV = Path('outputs/history/european_coverage.csv')
GAP_THRESHOLD = 7


def _collect():
    """Devuelve lista de dicts con info de cobertura por ticker."""
    rows = []
    providers = [
        ('Euronext', EuronextProvider()),
        ('Xetra', XetraProvider()),
        ('BME', BMEProvider()),
    ]
    today = pd.Timestamp.now().normalize()

    for source_name, provider in providers:
        for ticker in provider.supported_tickers():
            df = provider._load_cache(ticker)
            if df is None or df.empty:
                rows.append({
                    'ticker': ticker,
                    'source': source_name,
                    'last_date': None,
                    'days_gap': None,
                    'status': 'SIN_DATOS',
                })
                continue

            last_date = pd.to_datetime(df['date']).max()
            days_gap = int((today - last_date).days)
            status = 'REVISAR' if days_gap > GAP_THRESHOLD else 'OK'

            rows.append({
                'ticker': ticker,
                'source': source_name,
                'last_date': last_date,
                'days_gap': days_gap,
                'status': status,
            })
    return rows


def _render_markdown(rows, today_str):
    total = len(rows)
    ok = sum(1 for r in rows if r['status'] == 'OK')
    revisar = sum(1 for r in rows if r['status'] == 'REVISAR')
    sin_datos = sum(1 for r in rows if r['status'] == 'SIN_DATOS')

    lines = []
    lines.append('# Cobertura Europea - ' + today_str)
    lines.append('')
    lines.append('## Resumen')
    lines.append('- Tickers europeos monitoreados: ' + str(total))
    lines.append('- Cubiertos (gap <= ' + str(GAP_THRESHOLD) + ' dias): ' + str(ok))
    lines.append('- Con huecos > ' + str(GAP_THRESHOLD) + ' dias: ' + str(revisar))
    lines.append('- Sin datos en cache: ' + str(sin_datos))
    lines.append('')
    lines.append('## Detalle por ticker')
    lines.append('')
    lines.append('| Ticker | Fuente | Ultima fecha | Dias sin datos | Estado |')
    lines.append('|--------|--------|:------------:|:--------------:|:------:|')

    for r in sorted(rows, key=lambda x: (x['source'], x['ticker'])):
        last = r['last_date'].strftime('%Y-%m-%d') if r['last_date'] is not None else '-'
        gap = str(r['days_gap']) if r['days_gap'] is not None else '-'
        lines.append('| ' + r['ticker'] + ' | ' + r['source'] + ' | ' + last + ' | ' + gap + ' | ' + r['status'] + ' |')

    lines.append('')
    lines.append('## Tickers con huecos > ' + str(GAP_THRESHOLD) + ' dias (revisar)')
    lines.append('')
    revisar_rows = [r for r in rows if r['status'] == 'REVISAR']
    if not revisar_rows:
        lines.append('(vacio - sin alertas)')
    else:
        lines.append('| Ticker | Fuente | Ultima fecha | Dias sin datos |')
        lines.append('|--------|--------|:------------:|:--------------:|')
        for r in revisar_rows:
            lines.append('| ' + r['ticker'] + ' | ' + r['source'] + ' | ' + r['last_date'].strftime('%Y-%m-%d') + ' | ' + str(r['days_gap']) + ' |')
    lines.append('')
    lines.append('## Tickers sin datos en cache')
    lines.append('')
    sin_datos_rows = [r for r in rows if r['status'] == 'SIN_DATOS']
    if not sin_datos_rows:
        lines.append('(vacio - todos tienen cache)')
    else:
        lines.append('| Ticker | Fuente |')
        lines.append('|--------|--------|')
        for r in sin_datos_rows:
            lines.append('| ' + r['ticker'] + ' | ' + r['source'] + ' |')
    lines.append('')
    lines.append('---')
    lines.append('*Generado automaticamente por src/european_coverage.py*')
    return '\n'.join(lines) + '\n'


def _append_csv(rows, today_str):
    if not rows:
        return
    new_rows = []
    for r in rows:
        new_rows.append({
            'date': today_str,
            'ticker': r['ticker'],
            'source': r['source'],
            'last_date': r['last_date'].strftime('%Y-%m-%d') if r['last_date'] is not None else '',
            'days_gap': r['days_gap'] if r['days_gap'] is not None else '',
            'status': r['status'],
        })
    df_new = pd.DataFrame(new_rows)

    if OUTPUT_CSV.exists():
        try:
            df_old = pd.read_csv(OUTPUT_CSV)
            df = pd.concat([df_old, df_new], ignore_index=True)
            df = df.drop_duplicates(subset=['date', 'ticker'], keep='last')
            df = df.sort_values(['date', 'source', 'ticker']).reset_index(drop=True)
        except Exception:
            df = df_new
    else:
        df = df_new

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)


def generate_european_coverage_report():
    """Genera markdown + anexa CSV historico. Retorna dict resumen."""
    today = pd.Timestamp.now().normalize()
    today_str = today.strftime('%Y-%m-%d')

    rows = _collect()

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    md_content = _render_markdown(rows, today_str)
    OUTPUT_MD.write_bytes(md_content.encode('utf-8'))

    _append_csv(rows, today_str)

    ok = sum(1 for r in rows if r['status'] == 'OK')
    revisar = sum(1 for r in rows if r['status'] == 'REVISAR')
    sin_datos = sum(1 for r in rows if r['status'] == 'SIN_DATOS')

    print('  Cobertura europea: ' + str(ok) + ' OK, ' + str(revisar) + ' REVISAR, ' + str(sin_datos) + ' SIN_DATOS')
    print('  Reporte: ' + str(OUTPUT_MD))

    return {'total': len(rows), 'ok': ok, 'revisar': revisar, 'sin_datos': sin_datos}
