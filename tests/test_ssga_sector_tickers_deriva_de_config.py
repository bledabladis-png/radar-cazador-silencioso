# -*- coding: utf-8 -*-
"""FASE1 (2026-11-01): SECTOR_TICKERS de ssga deriva de MARKET_TICKERS.

Cierra el patron 15 de 01b_PATRONES.md (fuente de verdad duplicada).
Antes era lista literal; si config.tickers cambiaba, SSGA descargaba
los ETFs viejos silenciosamente. Ahora se deriva de config + FEZ
(FEZ esta fuera de MARKET_TICKERS['sectors'], es un ETF europeo).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.tickers import MARKET_TICKERS
from data.providers import ssga_fund_data


def test_sector_tickers_deriva_de_config():
    expected = set(MARKET_TICKERS["sectors"]) | {"FEZ"}
    actual = set(ssga_fund_data.SECTOR_TICKERS)
    assert actual == expected, (
        "SECTOR_TICKERS no coincide con config + FEZ. "
        "Solo en ssga: " + str(sorted(actual - expected)) + ". "
        "Solo en config+FEZ: " + str(sorted(expected - actual)) + "."
    )


def test_ssga_importa_market_tickers():
    """Anti-regresion: el modulo debe importar MARKET_TICKERS (no literal)."""
    src = (ROOT / "data" / "providers" / "ssga_fund_data.py").read_text(
        encoding="utf-8"
    )
    assert "from config.tickers import MARKET_TICKERS" in src, (
        "ssga_fund_data.py no importa MARKET_TICKERS. "
        "Reintroduce fuente duplicada."
    )
    assert "list(MARKET_TICKERS['sectors']) + ['FEZ']" in src, (
        "SECTOR_TICKERS no deriva de MARKET_TICKERS. "
        "Reintroduce lista literal."
    )
