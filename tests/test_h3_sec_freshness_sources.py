"""Test H3: la tabla de frescura declara las DOS fuentes SEC separadas.

Auditoria externa 2026-10-15. Antes se declaraba solo SEC N-PORT
(posiciones, trimestral, latencia ~60d), pero el pipeline consume
ademas N-PORT-P QQQ (flujos, cuadro B.6). El reporte daba impresion
de que SEC entera estaba stale cuando solo lo estaban las posiciones.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicators.data_quality import SOURCES


def _sources_sec():
    return [s for s in SOURCES if 'SEC' in s.get('source', '')]


def test_dos_fuentes_sec_declaradas():
    """Debe haber al menos 2 fuentes SEC (posiciones + flujos QQQ)."""
    sec = _sources_sec()
    assert len(sec) >= 2, (
        f"Solo {len(sec)} fuentes SEC declaradas. H3 exige separar "
        f"posiciones N-PORT de flujos N-PORT-P QQQ."
    )


def test_sec_posiciones_declarada():
    sec = _sources_sec()
    paths = [s.get('file', '') for s in sec]
    assert any('sec_nport_position_change_quarterly.csv' in p for p in paths), (
        "Falta la fuente 'SEC N-PORT (posiciones)'."
    )


def test_sec_flujos_qqq_declarada():
    sec = _sources_sec()
    paths = [s.get('file', '') for s in sec]
    assert any('qqq_nport_flow.csv' in p for p in paths), (
        "Falta la fuente 'SEC N-PORT-P QQQ (flujos)'. H3."
    )


def test_sec_fuentes_tienen_date_cols_correctas():
    sec = _sources_sec()
    for s in sec:
        src = s.get('source', '')
        cols = s.get('date_cols', [])
        if 'posiciones' in src:
            assert 'REPORT_DATE' in cols, (
                f"{src} debe declarar date_cols=['REPORT_DATE']"
            )
        if 'QQQ' in src or 'flujos' in src:
            assert 'report_date' in cols, (
                f"{src} debe declarar date_cols=['report_date']"
            )
