"""Tests del FinraProvider (F5.7-05). Sin red."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from data.providers.finra import FinraProvider, _LATEST_WEEK_TTL


def test_get_latest_week_memoiza():
    """La segunda llamada dentro del TTL no recalcula."""
    fp = FinraProvider()
    fake_df = pd.DataFrame({"issueSymbolIdentifier": ["AAPL"],
                             "totalWeeklyShareQuantity": [1000]})
    with patch.object(fp, "get_week_summary", return_value=fake_df) as m:
        r1 = fp.get_latest_week()
        r2 = fp.get_latest_week()
        r3 = fp.get_latest_week()
    assert r1 == r2 == r3
    assert m.call_count == 1, f"memoize no aplica: {m.call_count} llamadas"


def test_get_latest_week_ttl_expirado_recalcula():
    """Tras el TTL, la proxima llamada recalcula."""
    fp = FinraProvider()
    fake_df = pd.DataFrame({"issueSymbolIdentifier": ["AAPL"],
                             "totalWeeklyShareQuantity": [1000]})
    with patch.object(fp, "get_week_summary", return_value=fake_df) as m:
        fp.get_latest_week()
        fp._latest_week_ts -= (_LATEST_WEEK_TTL + 100)  # simular TTL vencido
        fp.get_latest_week()
    assert m.call_count == 2


def test_get_latest_week_sin_datos_memoiza_none():
    """Si no hay datos, memoiza None (no reintenta dentro del TTL)."""
    fp = FinraProvider()
    empty = pd.DataFrame()
    with patch.object(fp, "get_week_summary", return_value=empty) as m:
        r1 = fp.get_latest_week()
        r2 = fp.get_latest_week()
    assert r1 is None and r2 is None
    # 6 iteraciones por llamada efectiva x 1 llamada = 6
    assert m.call_count == 6


def test_cache_separada_por_instancia():
    """Dos instancias no comparten memoize."""
    fake_df = pd.DataFrame({"issueSymbolIdentifier": ["AAPL"],
                             "totalWeeklyShareQuantity": [1000]})
    fp1 = FinraProvider()
    fp2 = FinraProvider()
    with patch.object(fp1, "get_week_summary", return_value=fake_df) as m1, \
         patch.object(fp2, "get_week_summary", return_value=fake_df) as m2:
        fp1.get_latest_week()
        fp2.get_latest_week()
    assert m1.call_count == 1
    assert m2.call_count == 1


def test_is_available_sigue_existiendo():
    """is_available es contrato abstracto de MarketDataProvider."""
    fp = FinraProvider()
    assert hasattr(fp, "is_available")
    assert callable(fp.is_available)


def test_is_available_reutiliza_memoize():
    """Tras get_latest_week, is_available no dispara nueva busqueda."""
    fp = FinraProvider()
    fake_df = pd.DataFrame({"issueSymbolIdentifier": ["AAPL"],
                             "totalWeeklyShareQuantity": [1000]})
    with patch.object(fp, "get_week_summary", return_value=fake_df) as m:
        fp.get_latest_week()
        r = fp.is_available()
    assert r is True
    assert m.call_count == 1, f"is_available no reutiliza memoize: {m.call_count}"
