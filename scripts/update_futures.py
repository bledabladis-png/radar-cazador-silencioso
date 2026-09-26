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


def _classify_error(exc) -> str:
    """Clasifica el error para decidir exit code.

    F3-17: best-effort no debe ocultar errores permanentes (credenciales
    invalidas, plan sin acceso, limite agotado). Transitorios (timeout,
    red, 5xx) siguen siendo best-effort.

    Returns:
      "permanent" -> exit != 0. El workflow debe alertar.
      "transient" -> exit 0. El cron reintenta en el siguiente slot.
    """
    msg = str(exc).lower()
    permanent_markers = (
        "401", "unauthorized",
        "403", "feature access", "required_addon", "required_feature",
        "429", "too many",
    )
    for marker in permanent_markers:
        if marker in msg:
            return "permanent"
    return "transient"


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

        # F3-17 extension: detectar fallo silencioso del provider.
        # fetch_commodities captura internamente los 403/429 y devuelve
        # DataFrame vacio. main() ve "sin datos nuevos" sin excepcion.
        # Si habia futuros/spot esperados y no se recuperaron -> exit 1.
        fut_expected = bool(fut_missing)
        spot_expected = bool(spot_missing) and not spot_ok
        if fut_expected and not fut_res:
            print('update_futures: futures esperados pero no recuperados -> exit 1')
            return 1
        if spot_expected and not spot_res:
            print('update_futures: spot esperados pero no recuperados -> exit 1')
            return 1
        return 0
    except Exception as e:
        kind = _classify_error(e)
        print(f'update_futures: ERROR ({kind}) {e}')
        return 1 if kind == "permanent" else 0


if __name__ == '__main__':
    sys.exit(main())
