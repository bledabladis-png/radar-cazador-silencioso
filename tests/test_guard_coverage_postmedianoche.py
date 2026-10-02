"""Tests regresion: PENDING post-medianoche CEST (2026-10-02).

Bug: run 36942867498 (01:50 CEST del 2-oct). _compute_by_market marcaba
INVALID para mercados europeos con coverage 0, porque is_pending
comparaba last_session (2026-10-01) contra reference_date.date()
(2026-10-02). Ademas MARKETS_WITH_PUBLICATION_LAG solo contenia BME.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.utils import _compute_by_market
from src.market_hours import MARKETS_WITH_PUBLICATION_LAG

MADRID = ZoneInfo("Europe/Madrid")


def _make_df(markets_tickers, n_rows=5, end_date="2026-10-01"):
    idx = pd.date_range(end=end_date, periods=n_rows, freq="B")
    data = {}
    for market, tickers in markets_tickers.items():
        for t in tickers:
            data[("Close", t)] = [100.0] * n_rows
    return pd.DataFrame(data, index=idx)


def test_mercados_europeos_en_lista_de_lag():
    """Regresion 2026-10-02: los 4 europeos tienen lag de publicacion."""
    for m in ("BME", "EURONEXT", "LSE", "XETRA"):
        assert m in MARKETS_WITH_PUBLICATION_LAG, m + " debe estar en la lista"


def test_pending_post_medianoche_cest(monkeypatch):
    """Run a las 01:50 CEST del 2-oct: europeos con coverage 0 -> PENDING."""
    from src import instrument_registry as ir
    from src import market_hours as mh

    monkeypatch.setattr(ir, "get_market", lambda t: "EURONEXT")
    # Mock realista: a las 01:50 CEST del 2-oct, el cierre del 2-oct NO ha
    # pasado (18:00 CEST). Solo dias ANTERIORES estan cerrados.
    monkeypatch.setattr(mh, "is_session_closed",
                        lambda market, sd, ref: sd.weekday() < 5 and sd < ref.date())
    monkeypatch.setattr(mh, "is_trading_session", lambda m, d: d.weekday() < 5)

    # df termina el 2026-09-30 (Yahoo no ha publicado el 1-oct)
    idx = pd.date_range(end="2026-09-30", periods=5, freq="B")
    df = pd.DataFrame({("Close", "ASML"): [100.0] * 5}, index=idx)

    # reference_date = 2026-10-02 01:50 CEST. Sesion esperada = 2026-10-01.
    ref = datetime(2026, 10, 2, 1, 50, tzinfo=MADRID)
    result = _compute_by_market(df, ref)

    info = result["EURONEXT"]
    assert info["n"] == 1
    assert info["coverage_at_session"] == 0
    assert info["status"] == "PENDING", (
        "EURONEXT debe ser PENDING (lag Yahoo), no INVALID"
    )


def test_guard_exime_cuando_europeos_pending(monkeypatch):
    """_all_markets_valid devuelve True si todos son VALID o PENDING."""
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    from scripts.guard_coverage import _all_markets_valid

    by_market = {
        "BME": {"n": 19, "status": "PENDING", "coverage_at_session": 0.0},
        "EURONEXT": {"n": 13, "status": "PENDING", "coverage_at_session": 0.0},
        "LSE": {"n": 20, "status": "PENDING", "coverage_at_session": 0.0},
        "XETRA": {"n": 19, "status": "PENDING", "coverage_at_session": 0.0},
        "US_EQUITY": {"n": 245, "status": "VALID", "coverage_at_session": 1.0},
    }
    assert _all_markets_valid(by_market, 0.95) is True


def test_guard_no_exime_si_algun_europeo_sigue_invalid(monkeypatch):
    """Si algún europeo no es PENDING ni VALID, no exime."""
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))
    from scripts.guard_coverage import _all_markets_valid

    by_market = {
        "BME": {"n": 19, "status": "PENDING", "coverage_at_session": 0.0},
        "EURONEXT": {"n": 13, "status": "INVALID", "coverage_at_session": 0.0},
        "LSE": {"n": 20, "status": "PENDING", "coverage_at_session": 0.0},
        "US_EQUITY": {"n": 245, "status": "VALID", "coverage_at_session": 1.0},
    }
    assert _all_markets_valid(by_market, 0.95) is False