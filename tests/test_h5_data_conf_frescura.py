"""Test H5: Data Conf se deriva SOLO de frescura (no de cobertura),
y los umbrales se publican en el reporte.

Auditoria externa 2026-10-15, hallazgo H5 (alto): la nota del header
decia "Data Conf mide la frescura y cobertura", pero el codigo de
freshness.py solo usa frescura (CURRENT/RECENT -> Alta). Ademas, los
umbrales FRESHNESS_* no se publicaban.
"""
import pandas as pd

from src.report.header import render_regimenes
from src.report.freshness import render_data_freshness


def test_header_nota_no_dice_cobertura():
    """La nota del header NO debe afirmar que Data Conf mide cobertura."""
    out = render_regimenes(
        macro_score=pd.Series([0.0]), macro_regime="MIXED", macro_conf=0.5,
        liquidity_score=pd.Series([0.0]), financial_regime="NEUTRAL", liq_conf=0.5,
        volatility_score=pd.Series([0.0]), vol_regime="NORMAL", vol_conf=0.5,
        real_liquidity_regime="NEUTRAL", real_liquidity_conf=0.9,
        real_liq_score=pd.Series([0.0]), real_liq_prev=0.0,
        sector_regime="ROTATIONAL",
    )
    joined = "".join(out)
    assert "Data Conf" in joined
    # No debe decir "frescura y cobertura"
    assert "frescura y cobertura" not in joined.lower()
    # Debe decir que se deriva de la frescura
    assert "frescura observada" in joined.lower() or "frescura" in joined.lower()


def test_freshness_publica_umbrales():
    """La tabla compacta debe publicar los umbrales FRESHNESS_*."""
    out = render_data_freshness(
        pcr_data={"last_date": "2026-10-13", "zscore": 0.0},
        darkpool_data={"week": "2026-09-21"},
        sector_results={"ranking": [("XLK", "Technology", 0.4)]},
        reference_date=pd.Timestamp("2026-10-15"),
    )
    joined = "".join(out)
    assert "Umbrales de frescura" in joined or "umbrales" in joined.lower()
    # Al menos una de las tuplas debe aparecer
    assert "CBOE" in joined and "FINRA" in joined
    assert "CURRENT" in joined and "RECENT" in joined and "STALE" in joined
    assert "ARCHIVAL" in joined
