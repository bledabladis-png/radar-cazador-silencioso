"""Actualiza commodities via OilPriceAPI (FU-021-3C-bis).

Entry point del step 'Update commodities' en daily_run.yml.

Politica best-effort: si OilPriceAPI falla, exit 0. El pipeline sigue;
los contratos FUTURE_SETTLEMENT / SPOT_COMMODITY degradan a STALE.

Skip si los parquets ya estan al dia (date_max >= ultimo dia esperado).
Evita quemar requests en runs manuales repetidos.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

# Anadir root del repo al sys.path para permitir 'import data.*'
# cuando el script se ejecuta directamente (py scripts/update_futures.py).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.providers.futures import (  # noqa: E402
    FUTURES_MAP,
    SPOT_MAP,
    FuturesProvider,
)


FUTURES_PATH = 'data/commodities_futures.parquet'
SPOT_PATH = 'data/commodities_spot.parquet'


def _expected_settlement_date(reference_date):
    """Ultimo dia laborable esperado (proxy de settlement)."""
    try:
        from src.market_calendar import last_expected_market_date
        d = last_expected_market_date(reference_date)
        return pd.Timestamp(d).date() if d is not None else None
    except Exception:
        return None


def _inspect_parquet(path, expected_date, required_tickers):
    """Inspecciona el parquet y devuelve (is_complete, missing_tickers).

    K-FUTURES-REFRESH-01: el skip logic antiguo solo verificaba last_date,
    proxy de completitud. Un parquet con last_date == expected_date puede
    tener tickers con Close=NaN (fila parcial). El consumidor
    FUTURE_SETTLEMENT exige cobertura completa. Este helper cierra el gap.

    is_complete=True  <=>  last_date == expected_date
                            AND todos los required_tickers tienen Close
                            no-NaN en esa fila.
    missing_tickers   = required_tickers sin Close valido en esa fecha.
    """
    required = list(required_tickers)
    if expected_date is None:
        return False, required
    try:
        if not Path(path).exists():
            return False, required
        df = pd.read_parquet(path)
        if df is None or df.empty:
            return False, required
        last_ts = df.index.max()
        last = pd.Timestamp(last_ts).date()
        if last != expected_date:
            return False, required
        missing = []
        for t in required:
            col = ('Close', t)
            if col not in df.columns:
                missing.append(t)
                continue
            val = df.loc[last_ts, col]
            if pd.isna(val):
                missing.append(t)
        return len(missing) == 0, missing
    except Exception:
        return False, required


def main():
    reference_date = datetime.now(ZoneInfo('Europe/Madrid'))
    run_id = reference_date.strftime('%Y%m%d_%H%M%S')

    expected = _expected_settlement_date(reference_date)
    print(f'update_futures: reference={reference_date.date()} '
          f'run_id={run_id} expected_settlement={expected}')

    fut_ok, fut_missing = _inspect_parquet(
        FUTURES_PATH, expected, list(FUTURES_MAP.keys()))
    spot_ok, spot_missing = _inspect_parquet(
        SPOT_PATH, expected, list(SPOT_MAP.keys()))

    if fut_ok and spot_ok:
        print('update_futures: parquets ya al dia y completos. Skip fetch.')
        return 0

    print(f'update_futures: fetch. fut_missing={fut_missing} '
          f'spot_missing={spot_missing}')

    try:
        provider = FuturesProvider()
        result = provider.fetch_and_write(
            reference_date=reference_date,
            run_id=run_id,
            futures_path=FUTURES_PATH,
            spot_path=SPOT_PATH,
            only_futures=fut_missing,
            skip_spot=spot_ok,
        )
        fut_res = bool(result.get('futures'))
        spot_res = bool(result.get('spot'))
        print(f'update_futures: futures={fut_res} spot={spot_res}')
        return 0
    except Exception as e:
        print(f'update_futures: ERROR {e}')
        return 0  # best-effort: no romper el pipeline


if __name__ == '__main__':
    sys.exit(main())
