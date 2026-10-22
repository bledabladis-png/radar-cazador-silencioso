"""Actualiza ^VIX3M via CBOE (FU-021-3D).

Entry point del step 'Update CBOE' en daily_run.yml.

Politica best-effort (F3-17-extendido, 2026-09-28): errores
transitorios (timeout, red, 5xx) -> exit 0. Errores permanentes
(401, 403, 429) -> exit 2. El pipeline sigue (continue-on-error
en el workflow) pero la alerta queda visible en el log. El
contrato VOLATILITY_INDEX degrada a INSUFFICIENT en cualquier caso.

Skip si el parquet ya esta al dia (date_max >= ultimo dia esperado).
Evita fetch innecesario en runs manuales repetidos.

Refs: FU-021-3D, R6, dictamen del auditor E5.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.providers.cboe_index import CboeIndexProvider  # noqa: E402


PARQUET_PATH = 'data/cboe_vix3m.parquet'


def _expected_session_date(reference_date):
    """Ultimo dia de sesion NYSE esperado."""
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


def _classify_error(exc) -> str:
    """Clasifica el error para decidir exit code.

    F3-17: best-effort no debe ocultar errores permanentes
    (credenciales invalidas, plan sin acceso, limite agotado).
    Transitorios (timeout, red, 5xx) siguen siendo best-effort.

    Returns:
      "permanent" -> exit 2. El workflow debe alertar.
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


def _post_filter_non_trading_days(parquet_path, reference_date, run_id):
    """Filtra filas con fechas no bursatiles del parquet CBOE.

    CBOE publica filas cuando NYSE esta cerrado (observacion del
    New Year en viernes: 2010-12-31, 2021-12-31). El parquet debe
    tener solo sesiones NYSE. Funcion idempotente: si no hay nada
    que limpiar, no-op. Si falla, no rompe el pipeline.
    """
    try:
        from pathlib import Path as _P
        from src.market_calendar import is_market_day
        from src.utils import write_artifact_with_manifest
        if not _P(parquet_path).exists():
            return
        df = pd.read_parquet(parquet_path)
        if not isinstance(df.index, pd.DatetimeIndex) or len(df) == 0:
            return
        mask = [is_market_day(d.date()) for d in df.index]
        if all(mask):
            return
        df_filtered = df[mask]
        n_removed = len(df) - len(df_filtered)
        print(f'update_cboe: post-filter -> {n_removed} filas no bursatiles eliminadas')
        write_artifact_with_manifest(
            df_filtered, parquet_path, 'cboe_vix3m',
            reference_date, run_id,
        )
    except Exception as e:
        print(f'update_cboe: [WARN] post-filter non-trading-days: {e}')


def main():
    reference_date = datetime.now(ZoneInfo('Europe/Madrid'))
    run_id = reference_date.strftime('%Y%m%d_%H%M%S')

    expected = _expected_session_date(reference_date)
    print(f'update_cboe: reference={reference_date.date()} '
          f'run_id={run_id} expected_session={expected}')

    _post_filter_non_trading_days(PARQUET_PATH, reference_date, run_id)

    if _already_up_to_date(PARQUET_PATH, expected):
        print('update_cboe: parquet ya al dia. Skip fetch.')
        return 0

    try:
        provider = CboeIndexProvider()
        result = provider.fetch_and_write(
            reference_date=reference_date,
            run_id=run_id,
            parquet_path=PARQUET_PATH,
        )
        _post_filter_non_trading_days(PARQUET_PATH, reference_date, run_id)

        ok = bool(result)
        print(f'update_cboe: ok={ok}')
        return 0
    except Exception as e:
        kind = _classify_error(e)
        print(f'update_cboe: ERROR ({kind}) {e}')
        return 2 if kind == 'permanent' else 0


if __name__ == '__main__':
    sys.exit(main())
