"""F6-1 (2026-09-28): umbrales FINRA ajustados.

F6-02 del consolidado: 26 dias se clasificaba como CURRENT con los
umbrales antiguos (30, 45, 60). Ahora (20, 28, 45): CURRENT <= 20,
RECENT <= 28, STALE <= 45, ARCHIVAL > 45.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import FRESHNESS_FINRA
from src.report.helpers import _classify_finra_freshness


def test_umbrales_actuales():
    assert FRESHNESS_FINRA == (20, 28, 45)


def test_26_dias_ya_no_es_current():
    """F6-02: 26 dias debe clasificarse como RECENT, no CURRENT."""
    assert _classify_finra_freshness(26) == "RECENT"


def test_20_dias_es_current():
    assert _classify_finra_freshness(20) == "CURRENT"


def test_21_dias_es_recent():
    assert _classify_finra_freshness(21) == "RECENT"


def test_28_dias_es_recent():
    assert _classify_finra_freshness(28) == "RECENT"


def test_29_dias_es_stale():
    assert _classify_finra_freshness(29) == "STALE"


def test_45_dias_es_stale():
    assert _classify_finra_freshness(45) == "STALE"


def test_46_dias_es_archival():
    assert _classify_finra_freshness(46) == "ARCHIVAL"