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

    F3-17-extendido (patron de update_futures.py): best-effort no
    debe ocultar errores permanentes (credenciales invalidas, plan
    sin acceso, limite agotado). Transitorios (timeout, red, 5xx)
    siguen siendo best-effort.

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


def main():
    reference_date = datetime.now(ZoneInfo('Europe/Madrid'))
    run_id = reference_date.strftime('%Y%m%d_%H%M%S')

    expected = _expected_session_date(reference_date)
    print(f'update_cboe: reference={reference_date.date()} '
          f'run_id={run_id} expected_session={expected}')

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
        ok = bool(result)
        print(f'update_cboe: ok={ok}')
        return 0
    except Exception as e:
        kind = _classify_error(e)
        print(f'update_cboe: ERROR ({kind}) {e}')
        return 2 if kind == 'permanent' else 0


if __name__ == '__main__':
    sys.exit(main())