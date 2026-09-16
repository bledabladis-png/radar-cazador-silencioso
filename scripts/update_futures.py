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

from data.providers.futures import FuturesProvider  # noqa: E402


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


def _already_up_to_date(path, expected_date):
    """True si el parquet ya cubre la fecha esperada."""
    if expected_date is None:
        return False
    try:
        if not Path(path).exists():
            return False
        df = pd.read_parquet(path)
        if df is None or df.empty:
            return False
        last = pd.Timestamp(df.index.max()).date()
        return last >= expected_date
    except Exception:
        return False


def main():
    reference_date = datetime.now(ZoneInfo('Europe/Madrid'))
    run_id = reference_date.strftime('%Y%m%d_%H%M%S')

    expected = _expected_settlement_date(reference_date)
    print(f'update_futures: reference={reference_date.date()} '
          f'run_id={run_id} expected_settlement={expected}')

    if _already_up_to_date(FUTURES_PATH, expected) and \
       _already_up_to_date(SPOT_PATH, expected):
        print('update_futures: parquets ya al dia. Skip fetch.')
        return 0

    try:
        provider = FuturesProvider()
        result = provider.fetch_and_write(
            reference_date=reference_date,
            run_id=run_id,
            futures_path=FUTURES_PATH,
            spot_path=SPOT_PATH,
        )
        fut_ok = bool(result.get('futures'))
        spot_ok = bool(result.get('spot'))
        print(f'update_futures: futures={fut_ok} spot={spot_ok}')
        return 0
    except Exception as e:
        print(f'update_futures: ERROR {e}')
        return 0  # best-effort: no romper el pipeline


if __name__ == '__main__':
    sys.exit(main())
