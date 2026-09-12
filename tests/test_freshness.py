# -*- coding: utf-8 -*-
"""
Tests de frescura de datos.

Capas:
  1. Unit tests de clasificadores (siempre corren, sin I/O).
  2. Integracion sobre datos reales (skipif si el fichero no existe).
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest

from src.report.helpers import (
    _classify_freshness,
    _classify_finra_freshness,
    _classify_fred_freshness,
)
from indicators.data_quality import classify_freshness
from src.stock_data_loader import _classify_ticker

BASE = Path(__file__).resolve().parents[1]


# ============================================================
# CAPA 1 - Unit tests de clasificadores
# ============================================================

def test_classify_freshness_boundaries():
    """daily: 3/7/14 dias."""
    assert _classify_freshness(0) == "CURRENT"
    assert _classify_freshness(3) == "CURRENT"
    assert _classify_freshness(4) == "RECENT"
    assert _classify_freshness(7) == "RECENT"
    assert _classify_freshness(8) == "STALE"
    assert _classify_freshness(14) == "STALE"
    assert _classify_freshness(15) == "ARCHIVAL"
    assert _classify_freshness(100) == "ARCHIVAL"


def test_classify_finra_boundaries():
    """finra: 30/45/60 dias (retraso regulatorio)."""
    assert _classify_finra_freshness(0) == "CURRENT"
    assert _classify_finra_freshness(30) == "CURRENT"
    assert _classify_finra_freshness(31) == "RECENT"
    assert _classify_finra_freshness(45) == "RECENT"
    assert _classify_finra_freshness(46) == "STALE"
    assert _classify_finra_freshness(60) == "STALE"
    assert _classify_finra_freshness(61) == "ARCHIVAL"


def test_classify_fred_boundaries():
    """fred: 30/60/90 dias."""
    assert _classify_fred_freshness(0) == "CURRENT"
    assert _classify_fred_freshness(30) == "CURRENT"
    assert _classify_fred_freshness(31) == "RECENT"
    assert _classify_fred_freshness(60) == "RECENT"
    assert _classify_fred_freshness(61) == "STALE"
    assert _classify_fred_freshness(90) == "STALE"
    assert _classify_fred_freshness(91) == "ARCHIVAL"


def test_classify_freshness_por_frecuencia_cftc():
    """CFTC: 45/90/120 dias."""
    assert classify_freshness(0, "cftc") == "CURRENT"
    assert classify_freshness(45, "cftc") == "CURRENT"
    assert classify_freshness(46, "cftc") == "RECENT"
    assert classify_freshness(90, "cftc") == "RECENT"
    assert classify_freshness(91, "cftc") == "STALE"
    assert classify_freshness(120, "cftc") == "STALE"
    assert classify_freshness(121, "cftc") == "ARCHIVAL"


def test_classify_freshness_por_frecuencia_sec():
    """SEC: 45/90/120 dias."""
    assert classify_freshness(45, "sec") == "CURRENT"
    assert classify_freshness(90, "sec") == "RECENT"
    assert classify_freshness(120, "sec") == "STALE"
    assert classify_freshness(121, "sec") == "ARCHIVAL"


def test_classify_freshness_por_frecuencia_finra():
    """FINRA: 30/45/60 dias."""
    assert classify_freshness(30, "finra") == "CURRENT"
    assert classify_freshness(45, "finra") == "RECENT"
    assert classify_freshness(60, "finra") == "STALE"
    assert classify_freshness(61, "finra") == "ARCHIVAL"


def test_classify_freshness_por_frecuencia_fred():
    """FRED: 30/60/90 dias."""
    assert classify_freshness(30, "fred") == "CURRENT"
    assert classify_freshness(60, "fred") == "RECENT"
    assert classify_freshness(90, "fred") == "STALE"
    assert classify_freshness(91, "fred") == "ARCHIVAL"


def test_classify_freshness_por_frecuencia_daily():
    """daily: 3/7/14 dias."""
    assert classify_freshness(3, "daily") == "CURRENT"
    assert classify_freshness(7, "daily") == "RECENT"
    assert classify_freshness(14, "daily") == "STALE"
    assert classify_freshness(15, "daily") == "ARCHIVAL"


def test_classify_freshness_nan():
    """age=NaN -> N/D."""
    assert classify_freshness(float("nan"), "daily") == "N/D"
    assert classify_freshness(pd.NA, "daily") == "N/D"


def test_classify_ticker_failed_casos_vacios():
    """_classify_ticker: df None/vacio, sin Close, todo NaN -> (FAILED, None).

    B1 (2026-09-12): la firma cambio a (status, reason). Casos basicos
    devuelven FAILED con reason None. El expected_session se pasa como
    precondicion pero no se usa en estos casos.
    """
    expected = pd.Timestamp("2026-01-02").date()
    assert _classify_ticker("AAPL", None, expected) == ("FAILED", None)
    assert _classify_ticker("AAPL", pd.DataFrame(), expected) == ("FAILED", None)
    assert _classify_ticker("AAPL", pd.DataFrame({"Open": [1, 2]}), expected) == ("FAILED", None)
    empty_close = pd.DataFrame({("Close", "AAPL"): [float("nan"), float("nan")]})
    assert _classify_ticker("AAPL", empty_close, expected) == ("FAILED", None)


def test_classify_ticker_ok_reciente():
    """_classify_ticker: dato de hoy -> (OK, None).

    B1 (2026-09-12): expected_session == ultima fecha del df. Sin NaN
    en close[-1], los chequeos B1 no disparan.
    """
    hoy = pd.Timestamp.now().normalize()
    df = pd.DataFrame(
        {("Close", "AAPL"): [100.0, 101.0]},
        index=[hoy - pd.Timedelta(days=1), hoy],
    )
    expected = df.index[-1].date()
    assert _classify_ticker("AAPL", df, expected) == ("OK", None)


def test_classify_ticker_partial():
    """_classify_ticker: 10 dias -> (PARTIAL, None).

    B1: expected_session se alinea con ultima fecha del df para no
    disparar EXPECTED_SESSION_ABSENT.
    """
    hoy = pd.Timestamp.now().normalize()
    df = pd.DataFrame(
        {("Close", "AAPL"): [100.0, 101.0]},
        index=[hoy - pd.Timedelta(days=11), hoy - pd.Timedelta(days=10)],
    )
    expected = df.index[-1].date()
    assert _classify_ticker("AAPL", df, expected) == ("PARTIAL", None)


def test_classify_ticker_stale():
    """_classify_ticker: 30 dias -> (STALE, None)."""
    hoy = pd.Timestamp.now().normalize()
    df = pd.DataFrame(
        {("Close", "AAPL"): [100.0, 101.0]},
        index=[hoy - pd.Timedelta(days=31), hoy - pd.Timedelta(days=30)],
    )
    expected = df.index[-1].date()
    assert _classify_ticker("AAPL", df, expected) == ("STALE", None)


def test_classify_ticker_ultimo_nan_failed():
    """_classify_ticker: ultimo NaN en sesion esperada -> DATA_ISSUE.

    B1 (2026-09-12): cambio semantico intencional. Antes este caso
    devolvia FAILED. Ahora es DATA_ISSUE + MISSING_CLOSE_EXPECTED_SESSION
    porque el NaN esta en la sesion esperada. Este es el caso B1 por
    excelencia.
    """
    hoy = pd.Timestamp.now().normalize()
    df = pd.DataFrame(
        {("Close", "AAPL"): [100.0, float("nan")]},
        index=[hoy - pd.Timedelta(days=1), hoy],
    )
    expected = df.index[-1].date()
    assert _classify_ticker("AAPL", df, expected) == (
        "DATA_ISSUE", "MISSING_CLOSE_EXPECTED_SESSION"
    )


# ============================================================
# CAPA 2 - Integracion sobre datos reales (skipif si no existen)
# ============================================================

MAX_DAILY_AGE = 4   # tolera fin de semana + 1 festivo
MAX_DAILY_AGE_EU = 5
MAX_DQ_AGE = 3


def _age_days(ts):
    return (pd.Timestamp.now() - pd.Timestamp(ts)).days


@pytest.mark.skipif(
    not (BASE / "data" / "market_data.parquet").exists(),
    reason="market_data.parquet no existe (CI fresco)",
)
def test_market_data_fresh():
    df = pd.read_parquet(BASE / "data" / "market_data.parquet")
    assert len(df) > 0, "market_data.parquet vacio"
    last = df.index[-1]
    age = _age_days(last)
    assert age <= MAX_DAILY_AGE, f"market_data stale: {age} dias (max {MAX_DAILY_AGE})"


@pytest.mark.skipif(
    not (BASE / "data" / "stock_prices.parquet").exists(),
    reason="stock_prices.parquet no existe (CI fresco)",
)
def test_stock_prices_fresh():
    df = pd.read_parquet(BASE / "data" / "stock_prices.parquet")
    assert len(df) > 0, "stock_prices.parquet vacio"
    last = df.index[-1]
    age = _age_days(last)
    assert age <= MAX_DAILY_AGE, f"stock_prices stale: {age} dias (max {MAX_DAILY_AGE})"


@pytest.mark.skipif(
    not (BASE / "data" / "stock_prices.parquet").exists(),
    reason="stock_prices.parquet no existe (CI fresco)",
)
def test_european_tickers_recent():
    """Al menos el 80% de los tickers europeos tiene datos recientes."""
    import re
    df = pd.read_parquet(BASE / "data" / "stock_prices.parquet")
    tickers = set(df.columns.get_level_values(1))
    eu_pat = re.compile(r"\.(PA|AS|MI|DE|MC)$")
    eu = [t for t in tickers if eu_pat.search(t)]
    if not eu:
        pytest.skip("sin tickers europeos en stock_prices.parquet")

    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=MAX_DAILY_AGE_EU)
    recientes = 0
    for t in eu:
        try:
            series = df[("Close", t)].dropna()
            if len(series) > 0 and series.index[-1] >= cutoff:
                recientes += 1
        except KeyError:
            continue

    ratio = recientes / len(eu)
    assert ratio >= 0.80, f"solo {recientes}/{len(eu)} ({ratio:.0%}) europeos recientes"


@pytest.mark.skipif(
    not (BASE / "outputs" / "history" / "data_quality.csv").exists(),
    reason="data_quality.csv no existe (CI fresco)",
)
def test_data_quality_recent():
    """La ultima ejecucion de data_quality debe ser reciente."""
    df = pd.read_csv(BASE / "outputs" / "history" / "data_quality.csv", parse_dates=["date"])
    last = df["date"].max()
    age = _age_days(last)
    assert age <= MAX_DQ_AGE, f"data_quality.csv stale: {age} dias (max {MAX_DQ_AGE})"


@pytest.mark.skipif(
    not (BASE / "outputs" / "history" / "data_quality.csv").exists(),
    reason="data_quality.csv no existe (CI fresco)",
)
def test_data_quality_sin_archival():
    """Ninguna fuente debe estar en ARCHIVAL en la ultima ejecucion."""
    df = pd.read_csv(BASE / "outputs" / "history" / "data_quality.csv", parse_dates=["date"])
    last_date = df["date"].max()
    ultima = df[df["date"] == last_date]
    archival = ultima[ultima["freshness"] == "ARCHIVAL"]
    assert archival.empty, (
        f"fuentes ARCHIVAL en {last_date.date()}: {archival['source'].tolist()}"
    )

# ============================================================
# CAPA 2b - Frescura por provider europeo (individual)
# ============================================================

@pytest.mark.skipif(
    not (BASE / "outputs" / "history" / "european_coverage.csv").exists(),
    reason="european_coverage.csv no existe (CI fresco)",
)
@pytest.mark.parametrize("source", ["Euronext", "Xetra", "BME"])
def test_european_provider_recent(source):
    """Cada provider europeo debe tener >=90% de tickers con status OK."""
    df = pd.read_csv(BASE / "outputs" / "history" / "european_coverage.csv")
    sub = df[df["source"] == source]
    if sub.empty:
        pytest.skip(f"sin filas para {source} en european_coverage.csv")
    ok = int((sub["status"] == "OK").sum())
    ratio = ok / len(sub)
    assert ratio >= 0.90, f"{source}: solo {ok}/{len(sub)} ({ratio:.0%}) OK"


@pytest.mark.skipif(
    not (BASE / "outputs" / "history" / "data_quality.csv").exists(),
    reason="data_quality.csv no existe (CI fresco)",
)
def test_data_quality_europeos_si_presentes():
    """Si data_quality.csv incluye Euronext/Xetra/BME, no deben estar ARCHIVAL."""
    df = pd.read_csv(BASE / "outputs" / "history" / "data_quality.csv", parse_dates=["date"])
    last_date = df["date"].max()
    ultima = df[df["date"] == last_date]
    for eu in ("Euronext", "Xetra", "BME"):
        sub = ultima[ultima["source"] == eu]
        if not sub.empty:
            freshness = sub.iloc[0]["freshness"]
            assert freshness != "ARCHIVAL", f"{eu} esta ARCHIVAL en {last_date.date()}"

# ============================================================
# CAPA 3 - Salvaguarda contra fuentes oficiales
# Estos tests requieren red. Si la fuente no responde, se saltan.
# Verifican que el historico contiene el ultimo dato publicado por la fuente.
# ============================================================


def _fuente_cboe_ultimo_dato():
    """Devuelve la fecha del ultimo dato publicado por CBOE, o None si no se puede."""
    try:
        from data.providers.cboe import CboeProvider
        cp = CboeProvider()
        if not cp.is_available():
            return None
        data = cp.get_options_data()
        if not data:
            return None
        return data.get('date')
    except Exception:
        return None


def _fuente_finra_ultima_semana():
    """Devuelve la ultima semana (lunes) disponible en FINRA, o None."""
    try:
        from data.providers.finra import FinraProvider
        fp = FinraProvider()
        return fp.get_latest_week()
    except Exception:
        return None


@pytest.mark.network
def test_cboe_pcr_al_dia():
    """El historico PCR debe contener la fecha mas reciente publicada por CBOE.
    Si la fuente tiene un dato mas nuevo que el CSV, hay que investigar.
    """
    csv_path = BASE / "outputs" / "history" / "pcr_history.csv"
    if not csv_path.exists():
        pytest.skip("pcr_history.csv no existe")

    fuente = _fuente_cboe_ultimo_dato()
    if fuente is None:
        pytest.skip("CBOE no responde o no devuelve datos")

    df = pd.read_csv(csv_path, parse_dates=["date"])
    ultimo_csv = pd.Timestamp(df["date"].max()).date()
    fuente_fecha = pd.Timestamp(fuente).date()

    assert fuente_fecha <= ultimo_csv, (
        f"CBOE tiene dato mas nuevo ({fuente_fecha}) que el CSV ({ultimo_csv}). "
        f"El pipeline deberia haberlo cogido en el proximo run."
    )


@pytest.mark.network
def test_finra_darkpool_al_dia():
    """El historico darkpool debe contener la semana mas reciente publicada por FINRA."""
    csv_path = BASE / "outputs" / "history" / "darkpool_history.csv"
    if not csv_path.exists():
        pytest.skip("darkpool_history.csv no existe")

    fuente = _fuente_finra_ultima_semana()
    if fuente is None:
        pytest.skip("FINRA no responde o no devuelve datos")

    df = pd.read_csv(csv_path, parse_dates=["week"])
    ultimo_csv = pd.Timestamp(df["week"].max()).date()
    fuente_fecha = pd.Timestamp(fuente).date()

    assert fuente_fecha <= ultimo_csv, (
        f"FINRA tiene semana mas nueva ({fuente_fecha}) que el CSV ({ultimo_csv})."
    )

# ============================================================
# CAPA 4 - Calendario de mercado
# ============================================================

from datetime import datetime as _dt
from src.market_calendar import (
    is_market_day,
    previous_market_day,
    last_expected_market_date,
)


def test_is_market_day_sabado_domingo():
    assert is_market_day(_dt(2026, 9, 12).date()) is False   # sabado
    assert is_market_day(_dt(2026, 9, 13).date()) is False   # domingo


def test_is_market_day_laborables():
    assert is_market_day(_dt(2026, 9, 11).date()) is True    # viernes
    assert is_market_day(_dt(2026, 9, 14).date()) is True    # lunes


def test_is_market_day_festivos_nyse():
    assert is_market_day(_dt(2026, 1, 1).date()) is False    # New Year
    assert is_market_day(_dt(2026, 11, 26).date()) is False  # Thanksgiving
    assert is_market_day(_dt(2026, 12, 25).date()) is False  # Christmas


def test_previous_market_day_desde_lunes():
    assert previous_market_day(_dt(2026, 9, 14).date()) == _dt(2026, 9, 11).date()


def test_previous_market_day_desde_lunes_post_festivo():
    # 30 nov 2026 es lunes; 27 nov viernes medio-dia (NYSE abierto) es el anterior
    assert previous_market_day(_dt(2026, 11, 30).date()) == _dt(2026, 11, 27).date()


def test_last_expected_sabado_madrugada():
    # Sabado 02:00 -> antes de PUBLISH_HOUR -> viernes anterior
    assert last_expected_market_date(_dt(2026, 9, 12, 2, 0)) == _dt(2026, 9, 11).date()


def test_last_expected_martes_tarde():
    # Martes 22:00 -> antes de PUBLISH_HOUR (23) -> lunes
    assert last_expected_market_date(_dt(2026, 9, 15, 22, 0)) == _dt(2026, 9, 14).date()


def test_last_expected_lunes_madrugada():
    # Lunes 05:00 -> antes de PUBLISH_HOUR -> viernes anterior
    assert last_expected_market_date(_dt(2026, 9, 14, 5, 0)) == _dt(2026, 9, 11).date()


def test_last_expected_post_thanksgiving():
    # Viernes 27 nov 2026 (medio dia NYSE) 10:00 -> antes de PUBLISH_HOUR -> miercoles 25 (Thanksgiving jueves)
    assert last_expected_market_date(_dt(2026, 11, 27, 10, 0)) == _dt(2026, 11, 25).date()


def test_last_expected_market_date_acepta_date_y_timestamp():
    """B5: acepta date y pd.Timestamp ademas de datetime.

    - pd.Timestamp es subclase de datetime -> misma ruta que datetime.
    - date puro -> asume hora 0, aplica lag de publicacion (conservador).
    """
    from datetime import date as _date

    # date puro -> siempre retrocede un dia natural (hour=0 < PUBLISH_HOUR)
    assert last_expected_market_date(_date(2026, 9, 14)) == _dt(2026, 9, 11).date()
    assert last_expected_market_date(_date(2026, 9, 15)) == _dt(2026, 9, 14).date()
    assert last_expected_market_date(_date(2026, 9, 12)) == _dt(2026, 9, 11).date()

    # pd.Timestamp con hora >= PUBLISH_HOUR -> sesion del mismo dia
    assert last_expected_market_date(pd.Timestamp('2026-09-14 23:30')) == _dt(2026, 9, 14).date()
    # pd.Timestamp con hora < PUBLISH_HOUR -> retrocede
    assert last_expected_market_date(pd.Timestamp('2026-09-15 22:00')) == _dt(2026, 9, 14).date()


# ============================================================
# FU-007 - _last_market_session
# ============================================================

def test_last_market_session_sabado():
    """Sabado 12/09 -> viernes 11/09."""
    from src.report.helpers import _last_market_session
    d = _last_market_session(pd.Timestamp('2026-09-12 22:00'))
    assert d.date() == _dt(2026, 9, 11).date()


def test_last_market_session_domingo():
    """Domingo 13/09 -> viernes 11/09."""
    from src.report.helpers import _last_market_session
    d = _last_market_session(pd.Timestamp('2026-09-13'))
    assert d.date() == _dt(2026, 9, 11).date()


def test_last_market_session_dia_bursatil_sin_cambio():
    """Viernes 11/09 -> 11/09 (no modifica)."""
    from src.report.helpers import _last_market_session
    d = _last_market_session(pd.Timestamp('2026-09-11'))
    assert d.date() == _dt(2026, 9, 11).date()


def test_last_market_session_festivo():
    """Labor Day 07/09 -> viernes 04/09."""
    from src.report.helpers import _last_market_session
    d = _last_market_session(pd.Timestamp('2026-09-07'))
    assert d.date() == _dt(2026, 9, 4).date()
