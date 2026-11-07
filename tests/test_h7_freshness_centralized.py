# -*- coding: utf-8 -*-
"""Centinela H7-completado (2026-11-07): todos los umbrales de
freshness en data_quality.classify_freshness deben provenir de
config.settings, no estar hardcodeados.

Contexto: H7 (2026-10-16) centralizo FINRA, SEC y CFTC. Quedaron
hardcodeadas 2 ramas:
- fred: (30, 60, 90) coincidente con FRESHNESS_FRED.
- else: (3, 7, 14) coincidente con FRESHNESS_DEFAULT.
Y CBOE caia en else (usaba umbrales de DEFAULT 3/7/14, no su
propio FRESHNESS_CBOE 3/5/10).

El bug latente: si settings cambia un umbral, la clasificacion
NO cambia en las ramas hardcodeadas. El reporte mentiria.

Fix: importar FRESHNESS_DEFAULT, FRESHNESS_YAHOO, FRESHNESS_CBOE,
FRESHNESS_FRED; ramas explicitas para fred/cboe/yahoo; else ->
DEFAULT. Sources Yahoo y CBOE con frequency propia (antes 'daily').
"""
from __future__ import annotations

from pathlib import Path

import pytest

import indicators.data_quality as dq
import config.settings as settings


# ---------------------------------------------------------------------------
# 1. Comportamiento funcional: si settings cambia, classify cambia
# ---------------------------------------------------------------------------

class TestSettingsMandatorios:

    def test_fred_usa_settings(self, monkeypatch):
        """FRESHNESS_FRED modificado -> classify lo respeta."""
        monkeypatch.setattr(settings, "FRESHNESS_FRED", (5, 10, 15))
        # Recargar dq para que re-importe (no: dq importa al cargar).
        # En su lugar, parchear la referencia en el modulo dq.
        monkeypatch.setattr(dq, "FRESHNESS_FRED", (5, 10, 15))
        assert dq.classify_freshness(5, "fred") == "CURRENT"
        assert dq.classify_freshness(6, "fred") == "RECENT"
        assert dq.classify_freshness(11, "fred") == "STALE"
        assert dq.classify_freshness(16, "fred") == "ARCHIVAL"

    def test_cboe_usa_settings(self, monkeypatch):
        """FRESHNESS_CBOE modificado -> classify lo respeta."""
        monkeypatch.setattr(dq, "FRESHNESS_CBOE", (1, 2, 3))
        assert dq.classify_freshness(1, "cboe") == "CURRENT"
        assert dq.classify_freshness(2, "cboe") == "RECENT"
        assert dq.classify_freshness(3, "cboe") == "STALE"
        assert dq.classify_freshness(4, "cboe") == "ARCHIVAL"

    def test_yahoo_usa_settings(self, monkeypatch):
        """FRESHNESS_YAHOO modificado -> classify lo respeta."""
        monkeypatch.setattr(dq, "FRESHNESS_YAHOO", (1, 3, 6))
        assert dq.classify_freshness(1, "yahoo") == "CURRENT"
        assert dq.classify_freshness(3, "yahoo") == "RECENT"
        assert dq.classify_freshness(6, "yahoo") == "STALE"
        assert dq.classify_freshness(7, "yahoo") == "ARCHIVAL"

    def test_daily_cae_a_default(self, monkeypatch):
        """FRESHNESS_DEFAULT modificado -> daily lo respeta."""
        monkeypatch.setattr(dq, "FRESHNESS_DEFAULT", (1, 2, 3))
        assert dq.classify_freshness(1, "daily") == "CURRENT"
        assert dq.classify_freshness(2, "daily") == "RECENT"
        assert dq.classify_freshness(3, "daily") == "STALE"
        assert dq.classify_freshness(4, "daily") == "ARCHIVAL"


# ---------------------------------------------------------------------------
# 2. Contrato estructural: constants importadas, sin literales hardcodeados
# ---------------------------------------------------------------------------

class TestSinHardcode:

    def test_importa_las_4_constantes(self):
        src = Path("indicators/data_quality.py").read_text(encoding="utf-8")
        for name in ("FRESHNESS_DEFAULT", "FRESHNESS_YAHOO",
                     "FRESHNESS_CBOE", "FRESHNESS_FRED",
                     "FRESHNESS_FINRA", "FRESHNESS_SEC",
                     "FRESHNESS_CFTC"):
            assert name in src, f"falta {name} en imports/uso de data_quality.py"

    @pytest.mark.parametrize("patron", [
        "if age_days <= 30: return 'CURRENT'",
        "if age_days <= 60: return 'RECENT'",
        "if age_days <= 90: return 'STALE'",
        "if age_days <= 3: return 'CURRENT'",
        "if age_days <= 7: return 'RECENT'",
        "if age_days <= 14: return 'STALE'",
    ])
    def test_sin_umbrales_hardcodeados(self, patron):
        src = Path("indicators/data_quality.py").read_text(encoding="utf-8")
        assert patron not in src, (
            f"umbral hardcodeado en data_quality.py: {patron!r}. "
            "Debe usar FRESHNESS_* de config.settings."
        )


# ---------------------------------------------------------------------------
# 3. Sources: Yahoo y CBOE con frequency propia
# ---------------------------------------------------------------------------

class TestSources:

    def _by_name(self):
        return {s["source"]: s for s in dq.SOURCES}

    def test_yahoo_usa_frequency_yahoo(self):
        assert self._by_name()["Yahoo Finance"]["frequency"] == "yahoo"

    def test_cboe_usa_frequency_cboe(self):
        assert self._by_name()["CBOE PCR"]["frequency"] == "cboe"

    def test_ssga_sigue_daily(self):
        """SSGA no tiene umbral propio; cae en DEFAULT (intencional)."""
        assert self._by_name()["SSGA ETF Flow"]["frequency"] == "daily"
