"""
Logica compartida de los providers BlackRock (DAXEX, ISF.L).

Los modulos `blackrock_fund_data.py` y `blackrock_isf_fund_data.py`
son wrappers finos que parametrizan las constantes por ETF
(URL, cache, output CSV, referer, label) y delegan aqui.

Refs: A5-70 (refactor 2026-09-27).
"""
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ._fund_flow_utils import fund_flow_robust_zscore_with_regime


MESES_ES = {
    'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'ago': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dic': 12
}

def parse_fecha_es(fecha_str):
    """Convierte '27 dic 2000' a pd.Timestamp."""
    if pd.isna(fecha_str):
        return None
    partes = str(fecha_str).strip().split()
    if len(partes) != 3:
        return None
    try:
        dia = int(partes[0])
        mes_str = partes[1].lower()
        anio = int(partes[2])
        mes = MESES_ES.get(mes_str)
        if mes is None:
            return None
        return pd.Timestamp(year=anio, month=mes, day=dia)
    except (ValueError, TypeError):
        return None

def download_fund_file(url, cache_file, referer, label):
    """Descarga el archivo de BlackRock si no existe o si tiene más de 23 horas."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    use_cache = cache_file.exists()
    if use_cache:
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime > timedelta(hours=23):
            use_cache = False

    if use_cache:
        print(f'  Usando caché para {label}')
        return True

    print(f'  Descargando {label} desde BlackRock...')
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Referer': referer,
    }
    try:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        if len(r.content) < 100000:
            raise ValueError('Archivo demasiado pequeño, posible bloqueo')
        cache_file.write_bytes(r.content)
        print(f'  Guardado en caché: {cache_file} ({len(r.content)} bytes)')
        return True
    except Exception as e:
        print(f'  Error descargando {label}: {e}')
        if cache_file.exists():
            print('  Usando caché existente pese al error.')
            return True
        return False

def parse_hist_sheet(file_path: Path):
    """Parsea el SpreadsheetML y extrae la hoja Histórico como DataFrame."""
    # Leer bytes y limpiar BOM múltiples
    raw = file_path.read_bytes()
    text = raw.decode('utf-8-sig', errors='ignore').lstrip('\ufeff')
    # Reemplazar BOM por nada
    text = text.replace('\ufeff', '')
    root = ET.fromstring(text)

    # Namespaces comunes
    ns_ss = 'urn:schemas-microsoft-com:office:spreadsheet'
    ns_attrs = {
        'Name': f'{{{ns_ss}}}Name',
        'Data': f'{{{ns_ss}}}Data',
    }

    hist_rows = []
    found = False

    for worksheet in root.iter():
        if not worksheet.tag.endswith('Worksheet'):
            continue
        name = worksheet.attrib.get(ns_attrs['Name']) or worksheet.attrib.get('Name')
        if name == 'Histórico':
            found = True
            for table in worksheet.iter():
                if not table.tag.endswith('Table'):
                    continue
                for row in table:
                    if not row.tag.endswith('Row'):
                        continue
                    row_data = []
                    for cell in row:
                        if not cell.tag.endswith('Cell'):
                            continue
                        data_el = cell.find(f'{{{ns_ss}}}Data')
                        if data_el is not None and data_el.text:
                            row_data.append(data_el.text.strip())
                        else:
                            row_data.append(None)
                    if row_data:
                        hist_rows.append(row_data)

    if not found:
        raise ValueError('Hoja Histórico no encontrada')

    if not hist_rows:
        raise ValueError('Hoja Histórico vacía')

    # Buscar fila de encabezados
    header_row_idx = None
    for i, row in enumerate(hist_rows):
        joined = ' '.join([str(c) for c in row if c])
        if 'NAV' in joined and ('Shares' in joined or 'Acciones' in joined):
            header_row_idx = i
            break

    if header_row_idx is None:
        raise ValueError('No se encontró cabecera con NAV y Shares')

    headers = hist_rows[header_row_idx]
    data = hist_rows[header_row_idx + 1:]

    # Crear DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Renombrar columnas relevantes
    col_map = {}
    for col in df.columns:
        col_str = str(col)
        col_lower = col_str.lower()
        if 'nav' in col_lower:
            col_map[col_str] = 'nav'
        elif 'shares' in col_lower:
            col_map[col_str] = 'shares_outstanding'
        elif 'total net assets' in col_lower or 'patrimonio' in col_lower:
            col_map[col_str] = 'total_net_assets'
        elif 'date' in col_lower or 'fecha' in col_lower or 'a día' in col_lower:
            col_map[col_str] = 'date'
    df = df.rename(columns=col_map)

    required = ['date', 'nav', 'shares_outstanding', 'total_net_assets']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Faltan columnas {missing}')

    df = df[required]
    df['date'] = df['date'].apply(parse_fecha_es)
    for col in ['nav','shares_outstanding','total_net_assets']:
        df[col] = pd.to_numeric(df[col].replace({'--': np.nan, '': np.nan}), errors='coerce')

    df = df.dropna(subset=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def get_blackrock_primary_flow(
    *,
    url: str,
    cache_file: Path,
    output_csv: Path,
    referer: str,
    label: str,
    force_download: bool = False,
) -> pd.DataFrame:
    """Descarga/procesa el histórico y devuelve la última fila con flujo."""
    if force_download:
        download_fund_file(url, cache_file, referer, label)
    else:
        if not download_fund_file(url, cache_file, referer, label):
            return pd.DataFrame()

    print('  Parseando hoja Histórico...')
    df = parse_hist_sheet(cache_file)

    df_flow = df.dropna(subset=['shares_outstanding']).copy()
    print(f'  Registros con shares_outstanding: {len(df_flow)}')

    df_flow['shares_change'] = df_flow['shares_outstanding'].diff()
    df_flow['estimated_flow_eur'] = df_flow['shares_change'] * df_flow['nav']
    df_flow['flow_pct_assets'] = df_flow['estimated_flow_eur'] / df_flow['total_net_assets']
    df_flow['flow_zscore'], df_flow['flow_zscore_regime'] = fund_flow_robust_zscore_with_regime(
        df_flow['flow_pct_assets'], window=120, min_periods=20,
    )
    df_flow['flow_5d'] = df_flow['estimated_flow_eur'].rolling(5).mean()
    df_flow['flow_20d'] = df_flow['estimated_flow_eur'].rolling(20).mean()

    cols = ['date','nav','shares_outstanding','shares_change','total_net_assets',
            'estimated_flow_eur','flow_pct_assets','flow_zscore','flow_zscore_regime','flow_5d','flow_20d']
    result = df_flow[cols]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f'  Guardado en {output_csv}')
    print(f'  Total filas: {len(result)}')
    print(result.tail(5).to_string(index=False))

    return result.tail(1)
