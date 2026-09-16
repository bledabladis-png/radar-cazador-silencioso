import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os
import time
from config.tickers import MARKET_TICKERS
from config.settings import CACHE_HOURS, CACHE_VALIDATE_TRADING_DATE
from data.providers.router import DataRouter
from data.providers.backup_providers import BackupProvider
from src.effective_date import resolve_effective_date
from src.market_hours import is_trading_session, is_session_closed
from src.instrument_registry import get_market

YAHOO_TICKER_MAP = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
    "MOGA": "MOG-A",
    "MOG A": "MOG-A",
    "GEF B": "GEF-B",
    "CRD A": "CRD-A",
    "BH A": "BH-A",
}

def normalize_yahoo_ticker(t):
    """Convierte tickers problemáticos al formato que acepta Yahoo Finance."""
    return YAHOO_TICKER_MAP.get(t, t)

def _ticker_list():
    tickers = []
    for group in MARKET_TICKERS.values():
        if isinstance(group, dict):
            tickers.extend(group.values())
        elif isinstance(group, list):
            tickers.extend(group)

    # Añadir tickers de los líderes sectoriales (etf_holdings.csv)
    try:
        import pandas as _pd
        holdings = _pd.read_csv('data/etf_holdings.csv')
        if 'ticker' in holdings.columns:
            # Lista negra de tickers inválidos detectados en holdings (futuros, CUSIP, efectivo)
            INVALID_TICKERS = {'XARU6','IXDU6','IXIU6','IXAU6','IXTU6','IXRU6',
                               'IXPU6','IXCU6','IXYU6','IXSU6','XASU6',
                               '2602335D','-'}
            raw_tickers = holdings['ticker'].tolist()
            for t in raw_tickers:
                if isinstance(t, str) and t not in INVALID_TICKERS:
                    tickers.append(normalize_yahoo_ticker(t))
    except Exception as e:
        print(f"  [WARN] _ticker_list: etf_holdings.csv: {e}")

    return list(set(tickers))

# Nota: sin @retry global. El bucle por lotes ya gestiona fallos
# y BackupProvider actua como fallback por lote.
def _is_equity_ticker(t):
    """FU-021-3A (2026-09-15): True si el ticker es equity/ETF USA.

    Excluye:
    - indices (^): ^GSPC, ^VIX, ^TNX, ^FVX, ^FTSE, ^GDAXI, ^IBEX, ^STOXX50E, ^SPGSCI
    - futuros (=F): CL=F, BZ=F, NG=F, GC=F, HG=F
    - FX (=X): EURUSD=X, USDJPY=X, USDCNY=X
    - DXY (DX-Y.NYB): indice de divisas ICE, clasificado como INDEX_EOD
    """
    s = str(t)
    return (not s.startswith('^')
            and not s.endswith('=F')
            and not s.endswith('=X')
            and s != 'DX-Y.NYB')


def _filter_non_eod_equity(data, reference_date):
    """FU-021-3A correction v2 (2026-09-15): aplica el mecanismo FU-018
    (sesion/EOD) SOLO al universo EQUITY_EOD dentro de market_data.

    Politica conservadora: si ALGUN ticker equity con dato en la ultima
    fecha tiene observacion no EOD (mercado aun abierto o mercado
    desconocido), se elimina la ultima fila completa del DataFrame,
    incluyendo columnas no-equity de esa fila. Evita filas hibridas.

    Los tickers no-equity (INDEX_EOD, RATE_YIELD, FUTURE_SETTLEMENT,
    FX_DAILY_CUT) NO se inspeccionan aqui: su contrato temporal queda
    pendiente de 3B/3C segun FU-021-2.

    reference_date debe ser tz-aware. run.py:main() lo garantiza.

    Devuelve (data_posiblemente_recortado, info).
    """
    info = {'n_equity': 0, 'n_present': 0, 'non_eod': 0, 'last_date': None}
    if data is None or data.empty or reference_date is None:
        return data, info

    equity_tickers = [
        c[1] for c in data.columns
        if len(c) == 2 and c[0] == 'Close' and _is_equity_ticker(c[1])
    ]
    info['n_equity'] = len(equity_tickers)
    if not equity_tickers:
        return data, info

    last_date = pd.Timestamp(data.index[-1]).date()
    info['last_date'] = last_date

    present = [t for t in equity_tickers
               if ('Close', t) in data.columns
               and pd.notna(data[('Close', t)].iloc[-1])]
    info['n_present'] = len(present)
    if not present:
        return data, info

    non_eod = 0
    for t in present:
        market = get_market(t)
        if market == 'UNKNOWN':
            non_eod += 1
            continue
        if not is_trading_session(market, last_date):
            continue
        if not is_session_closed(market, last_date, reference_date):
            non_eod += 1
    info['non_eod'] = non_eod

    if non_eod == 0:
        return data, info

    print(f"  [FU-021-3A] EQUITY_EOD: {non_eod}/{len(present)} equity con "
          f"vela no EOD en {last_date} -> eliminar ultima fila")
    return data.iloc[:-1], info


def _trim_market_data_to_equity_eod(data):
    """FU-021-3A: recorta data a la ultima fecha con cobertura EQUITY_EOD >= 90%.

    Devuelve (data_recortado, meta) donde meta es el dict de
    resolve_effective_date (o None si no habia equities en el DataFrame).

    El universo elegible son unicamente los 539 tickers equity/ETF USA.
    INDEX_EOD, RATE_YIELD, FUTURE_SETTLEMENT y FX_DAILY_CUT quedan FUERA
    de este filtro (fases 3B y 3C). Cuando la fila superior se elimina,
    tambien se eliminan las columnas no-equity de esa fila (efecto
    colateral aceptado por auditor: no dejar filas hibridas).

    No lanza excepcion. Si no hay cobertura suficiente, no recorta y
    devuelve meta con status=INSUFFICIENT_COVERAGE.
    """
    _equity_tickers = [
        c[1] for c in data.columns
        if len(c) == 2 and c[0] == 'Close' and _is_equity_ticker(c[1])
    ]
    if not _equity_tickers:
        return data, None
    meta = resolve_effective_date(data, _equity_tickers, min_coverage=0.90)
    if (meta['status'] == 'OK'
            and meta['date'] is not None
            and meta['lag_days'] is not None
            and meta['lag_days'] > 0):
        data = data.loc[:meta['date']]
    return data, meta


def download_market_data(reference_date=None, run_id=None):
    # FU-002 (2026-09-15): reference_date y run_id inyectados desde run.py:main().
    if reference_date is None:
        reference_date = datetime.now(ZoneInfo("Europe/Madrid"))
    if run_id is None:
        run_id = reference_date.strftime('%Y%m%d_%H%M%S')
    cache_path = 'data/market_data.csv'
    parquet_path = 'data/market_data.parquet'
    # D3 Fase 2: elegir cache disponible (parquet o csv, el mas reciente)
    _candidates = []
    if os.path.exists(parquet_path):
        _candidates.append((parquet_path, 'parquet'))
    if os.path.exists(cache_path):
        _candidates.append((cache_path, 'csv'))
    if _candidates:
        _path, _fmt = max(_candidates, key=lambda t: os.path.getmtime(t[0]))
        mtime = datetime.fromtimestamp(os.path.getmtime(_path))
        if datetime.now() - mtime < timedelta(hours=CACHE_HOURS):
            _df = None
            if _fmt == 'parquet':
                try:
                    _df = pd.read_parquet(_path)
                except Exception as e:
                    print(f'  [WARN] Error leyendo Parquet: {e}')
            else:
                _df = pd.read_csv(_path, header=[0,1], index_col=0, parse_dates=True)
            # CACHE_VALIDATE_TRADING_DATE: verificar que el cache cubre
            # el ultimo dia de mercado esperado. Si no, forzar descarga.
            if _df is not None and CACHE_VALIDATE_TRADING_DATE and len(_df) > 0:
                from src.market_calendar import last_expected_market_date
                _last_exp = last_expected_market_date()
                _df_last = _df.index[-1].date() if hasattr(_df.index[-1], 'date') else _df.index[-1]
                if _df_last < _last_exp:
                    print(f'  [CACHE] datos hasta {_df_last}, esperado >= {_last_exp}. Forzando descarga.')
                else:
                    return _df
            elif _df is not None:
                return _df

    tickers = _ticker_list()
    router = DataRouter()
    backup = BackupProvider()

    # Descarga por lotes para evitar bloqueos
    batch_size = 5
    delay = 2
    all_data = []
    batches_failed = []   # lotes completos que fallaron
    batch_errors = []     # mensajes de error asociados a cada lote fallido

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"Descargando lote {i//batch_size + 1}: {batch}")
        try:
            data_batch = router.get_market_data(batch, period="10y")
            if data_batch is not None and not data_batch.empty:
                all_data.append(data_batch)
            else:
                # Si el lote devuelve vacío, registrar tickers
                batches_failed.append(batch)
                batch_errors.append('Datos vacíos')
        except Exception as e:
            print(f"Error en lote {batch}: {e}")
            try:
                backup_data = backup.get_prices(batch, period="10y")
                if backup_data is not None and not backup_data.empty:
                    all_data.append(backup_data)
                    print(f"  Respaldo obtuvo datos para {batch}")
                else:
                    batches_failed.append(batch)
                    batch_errors.append(str(e))
            except Exception as be:
                print(f"  Error en respaldo: {be}")
                batches_failed.append(batch)
                batch_errors.append(str(e))
        if i + batch_size < len(tickers):
            time.sleep(delay)

    if not all_data:
        raise RuntimeError("No se pudo descargar ningún ticker.")

    # --- Registrar tickers con fallos de descarga ---
    failed = []
    for batch in batches_failed:
        failed.extend(batch)
    if failed:
        # Crear directorio si no existe
        os.makedirs('outputs/audit', exist_ok=True)
        with open('outputs/audit/download_failures.md', 'w', encoding='utf-8') as f:
            f.write('# Fallos de descarga de tickers\n\n')
            f.write(f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            f.write('| Ticker | Lote | Error |\n')
            f.write('|--------|------|-------|\n')
            for i, batch in enumerate(batches_failed):
                for t in batch:
                    err_msg = batch_errors[i] if i < len(batch_errors) else 'Error desconocido'
                    f.write(f'| {t} | {i+1} | {err_msg} |\n')
        print(f'  Se registraron {len(failed)} tickers con fallos en outputs/audit/download_failures.md')
    else:
        # Si no hay fallos, borrar archivo anterior
        if os.path.exists('outputs/audit/download_failures.md'):
            os.remove('outputs/audit/download_failures.md')
    # ----------------------------------------------------

    data = pd.concat(all_data, axis=1)

    if not isinstance(data.columns, pd.MultiIndex):
        data.columns = pd.MultiIndex.from_tuples(data.columns)

    from src.utils import clean_oil_prices
    data = clean_oil_prices(data)

    # FU-021-3A correction v2 (2026-09-15): filtro sesion/EOD FU-018
    # aplicado SOLO al universo EQUITY_EOD. Ver FU-021-2 y dictamen v2.
    data, _eod_info = _filter_non_eod_equity(data, reference_date)

    # FU-021-3A (2026-09-15): filtro EQUITY_EOD.
    # Solo aplica al universo equity/ETF (539 tickers). No toca
    # INDEX_EOD / RATE_YIELD / FUTURE_SETTLEMENT / FX_DAILY_CUT.
    # Ver FU-021-2 (contrato por clase).
    data, _eff = _trim_market_data_to_equity_eod(data)
    if _eff and _eff['status'] == 'OK' and _eff['date'] is not None:
        _ref_date = (reference_date.date()
                     if hasattr(reference_date, 'date') else reference_date)
        _ref_lag = (pd.Timestamp(_ref_date)
                    - pd.Timestamp(_eff['date'].date())).days
        print(f"  [FU-021-3A] EQUITY_EOD effective={_eff['date'].date()} "
              f"requested={_eff['requested_date'].date()} "
              f"function_lag={_eff['lag_days']}d "
              f"reference_lag={_ref_lag}d "
              f"coverage={_eff['coverage']:.2%} "
              f"({_eff['n_observed']}/{_eff['n_eligible']})")
    elif _eff:
        print("  [FU-021-3A] EQUITY_EOD INSUFFICIENT_COVERAGE. Sin trim.")

    # FU-021-3C-bis: enriquecer con commodities (OilPriceAPI) si estan
    # disponibles. Solo merge en fechas comunes. No fetch aqui.
    from src.commodities_merge import merge_commodities_into_market
    data = merge_commodities_into_market(data)

    # FU-002 (2026-09-15): parquet + manifest atomico.
    from src.utils import write_artifact_with_manifest
    write_artifact_with_manifest(
        data, parquet_path,
        source='yahoo',
        reference_date=reference_date,
        run_id=run_id,
    )

    # FU-021-5 (Fase 3): resolver contratos y adjuntar temporal_meta.
    # Dictamen Q-P.3: dict paralelo es autoridad; df.attrs es espejo.
    try:
        from src.temporal_contracts import resolve_all_contracts
        from src.temporal_contracts.consolidate import build_temporal_meta
        _resolutions = resolve_all_contracts(data, reference_date)
        _meta = build_temporal_meta(_resolutions, reference_date, run_id)
        data.attrs['temporal_meta'] = _meta
        _summary = ' '.join(
            f"{k}={v.status}" for k, v in _resolutions.items()
        )
        print(f"  [FU-021-5] 9 contratos resueltos: {_summary}")
    except Exception as e:
        print(f"  [FU-021-5][WARN] Fallo resolviendo contratos: {e}")

    return data
