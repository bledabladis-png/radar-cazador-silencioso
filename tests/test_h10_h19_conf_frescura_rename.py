# -*- coding: utf-8 -*-
"""REP-AUDIT-20261017-H10/H19: la columna "Data Conf" del reporte se
renombra a "Conf. Frescura".

Motivacion: el auditor (2026-10-17) observo que "Data Conf" se
interpretaba como confianza global del dato. Pero solo expresa frescura
(CURRENT/RECENT -> Alta; STALE/ARCHIVAL -> Baja). Una fuente reciente
con cobertura 18% (Leader Representativeness) aparecia como "Alta", lo
cual sugiere una fiabilidad que no existe.

Fix: renombrar a "Conf. Frescura". La cobertura sigue siendo una
dimension independiente, reportada en la tabla Calidad.
"""
import pandas as pd

from src.report.header import render_regimenes
from src.report.freshness import render_data_freshness


def test_h10_columna_se_llama_conf_frescura():
    """La tabla Data Freshness tiene 'Conf. Frescura', no 'Data Conf'."""
    out = render_data_freshness(
        pcr_data={"last_date": "2026-10-13", "zscore": 0.0},
        darkpool_data={"week": "2026-09-21"},
        sector_results={"ranking": [("XLK", "Technology", 0.4)]},
        reference_date=pd.Timestamp("2026-10-15"),
    )
    joined = "".join(out)
    assert "Conf. Frescura" in joined
    assert "Data Conf" not in joined


def test_h10_header_usa_conf_frescura():
    out = render_regimenes(
        macro_score=pd.Series([0.0]), macro_regime="MIXED", macro_conf=0.5,
        liquidity_score=pd.Series([0.0]), financial_regime="NEUTRAL", liq_conf=0.5,
        volatility_score=pd.Series([0.0]), vol_regime="NORMAL", vol_conf=0.5,
        real_liquidity_regime="NEUTRAL", real_liquidity_conf=0.9,
        real_liq_score=pd.Series([0.0]), real_liq_prev=0.0,
        sector_regime="ROTATIONAL",
    )
    joined = "".join(out)
    assert "Conf. Frescura" in joined
    assert "Data Conf" not in joined


def test_h10_nota_aclara_independencia_de_cobertura():
    out = render_data_freshness(
        pcr_data={"last_date": "2026-10-13", "zscore": 0.0},
        darkpool_data={"week": "2026-09-21"},
        sector_results={"ranking": []},
        reference_date=pd.Timestamp("2026-10-15"),
    )
    joined = "".join(out)
    assert "NO de la cobertura" in joined or "no la cobertura" in joined or "independiente" in joined.lower()


def test_h10_valores_alta_baja_se_mantienen():
    out = render_data_freshness(
        pcr_data={"last_date": "2026-10-13", "zscore": 0.0},
        darkpool_data={"week": "2026-09-21"},
        sector_results={"ranking": []},
        reference_date=pd.Timestamp("2026-10-15"),
    )
    joined = "".join(out)
    assert ("Alta" in joined or "Baja" in joined or "N/D" in joined)


def test_h10_nota_umbrales_menciona_conf_frescura():
    out = render_data_freshness(
        pcr_data={"last_date": "2026-10-13", "zscore": 0.0},
        darkpool_data={"week": "2026-09-21"},
        sector_results={"ranking": []},
        reference_date=pd.Timestamp("2026-10-15"),
    )
    joined = "".join(out)
    assert "Conf. Frescura" in joined
    assert "Data Conf:" not in joined
