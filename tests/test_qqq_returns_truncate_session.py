# -*- coding: utf-8 -*-
"""Fix H: qqq_returns_yahoo trunca a la ultima sesion cerrada.

Bug: get_adjusted_prices usa prices.index[-1] sin filtrar. Yahoo
devuelve la barra parcial intradia si el mercado USA esta abierto.
El CSV publica effectiveDate=2026-09-30 a las 20:13 CEST con NYSE
abierto, mientras el resto del pipeline publica 2026-09-29.

Fix: get_adjusted_prices acepta reference_date opcional y pasa por
last_expected_market_date. Misma convencion que stock_data_loader,
data_loader, leaders.
"""
import importlib.util
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "qqq_returns_yahoo",
    str(ROOT / "scripts" / "qqq_returns_yahoo.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _make_prices(end_date: str, n: int = 2600):
    idx = pd.date_range(end=end_date, periods=n, freq="B")
    return pd.Series([100.0 + i * 0.1 for i in range(n)], index=idx)


def test_get_adjusted_prices_trunca_a_sesion_esperada():
    """Yahoo devuelve hasta 2026-09-30 (intradia). reference_date
    de 2026-09-30 20:00 CEST -> truncar a 2026-09-29."""
    prices_raw = _make_prices("2026-09-30")
    ref = datetime(2026, 9, 30, 20, 0, tzinfo=ZoneInfo("Europe/Madrid"))

    with patch.object(mod.yf, "download") as mock_dl:
        mock_dl.return_value = pd.DataFrame({"Close": prices_raw})
        prices = mod.get_adjusted_prices("QQQ", reference_date=ref)

    assert prices.index[-1] == pd.Timestamp("2026-09-29"), (
        f"Esperado 2026-09-29, obtenido {prices.index[-1]}. "
        f"Si es 2026-09-30, no esta truncando a last_expected_market_date."
    )


def test_get_adjusted_prices_no_trunca_si_ya_es_sesion_cerrada():
    """Si Yahoo devuelve hasta 2026-09-29 y reference_date avanza,
    no debe truncar mas alla."""
    prices_raw = _make_prices("2026-09-29")
    # reference_date de 2026-10-01 01:00 UTC -> ultima sesion = 2026-09-30
    # Yahoo ya solo tiene hasta 29, no debe tocar.
    ref = datetime(2026, 10, 1, 1, 0, tzinfo=ZoneInfo("Europe/Madrid"))

    with patch.object(mod.yf, "download") as mock_dl:
        mock_dl.return_value = pd.DataFrame({"Close": prices_raw})
        prices = mod.get_adjusted_prices("QQQ", reference_date=ref)

    assert prices.index[-1] == pd.Timestamp("2026-09-29")