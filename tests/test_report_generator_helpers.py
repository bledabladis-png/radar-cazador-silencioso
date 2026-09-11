"""
Tests unitarios de los helpers de src/report_generator.py.
Solo funciones puras, sin mocks ni dependencias externas.
"""
import pandas as pd
import numpy as np
import pytest

from src.report_generator import (
    _fmt_num,
    _classify_freshness,
    _classify_finra_freshness,
    _classify_fred_freshness,
    _generate_coverage_table,
)


# ============================================================
# _fmt_num
# ============================================================

class TestFmtNum:
    def test_valor_normal_2d(self):
        assert _fmt_num(3.14159) == "3.14"

    def test_valor_entero_con_fmt_0d(self):
        assert _fmt_num(42, "{:.0f}") == "42"

    def test_nan_devuelve_N_D(self):
        assert _fmt_num(float('nan')) == "N/D"

    def test_none_devuelve_N_D(self):
        assert _fmt_num(None) == "N/D"

    def test_numpy_nan_devuelve_N_D(self):
        assert _fmt_num(np.nan) == "N/D"

    def test_cero(self):
        assert _fmt_num(0.0) == "0.00"

    def test_negativo(self):
        assert _fmt_num(-1.5) == "-1.50"

    def test_formato_porcentaje(self):
        assert _fmt_num(0.125, "{:.1%}") == "12.5%"

    def test_formato_con_signo(self):
        assert _fmt_num(1.5, "{:+.2f}") == "+1.50"
        assert _fmt_num(-1.5, "{:+.2f}") == "-1.50"


# ============================================================
# _classify_freshness
# ============================================================

class TestClassifyFreshness:
    def test_cero_dias_current(self):
        assert _classify_freshness(0) == 'CURRENT'

    def test_3_dias_current(self):
        assert _classify_freshness(3) == 'CURRENT'

    def test_4_dias_recent(self):
        assert _classify_freshness(4) == 'RECENT'

    def test_7_dias_recent(self):
        assert _classify_freshness(7) == 'RECENT'

    def test_8_dias_stale(self):
        assert _classify_freshness(8) == 'STALE'

    def test_14_dias_stale(self):
        assert _classify_freshness(14) == 'STALE'

    def test_15_dias_archival(self):
        assert _classify_freshness(15) == 'ARCHIVAL'

    def test_30_dias_archival(self):
        assert _classify_freshness(30) == 'ARCHIVAL'

    def test_limites_personalizados(self):
        assert _classify_freshness(5, max_current=5) == 'CURRENT'
        assert _classify_freshness(6, max_current=5) == 'RECENT'


# ============================================================
# _classify_finra_freshness
# ============================================================

class TestClassifyFinraFreshness:
    def test_cero_current(self):
        assert _classify_finra_freshness(0) == 'CURRENT'

    def test_30_dias_current(self):
        assert _classify_finra_freshness(30) == 'CURRENT'

    def test_31_dias_recent(self):
        assert _classify_finra_freshness(31) == 'RECENT'

    def test_45_dias_recent(self):
        assert _classify_finra_freshness(45) == 'RECENT'

    def test_46_dias_stale(self):
        assert _classify_finra_freshness(46) == 'STALE'

    def test_60_dias_stale(self):
        assert _classify_finra_freshness(60) == 'STALE'

    def test_61_dias_archival(self):
        assert _classify_finra_freshness(61) == 'ARCHIVAL'


# ============================================================
# _classify_fred_freshness
# ============================================================

class TestClassifyFredFreshness:
    def test_30_dias_current(self):
        assert _classify_fred_freshness(30) == 'CURRENT'

    def test_31_dias_recent(self):
        assert _classify_fred_freshness(31) == 'RECENT'

    def test_60_dias_recent(self):
        assert _classify_fred_freshness(60) == 'RECENT'

    def test_61_dias_stale(self):
        assert _classify_fred_freshness(61) == 'STALE'

    def test_90_dias_stale(self):
        assert _classify_fred_freshness(90) == 'STALE'

    def test_91_dias_archival(self):
        assert _classify_fred_freshness(91) == 'ARCHIVAL'


# ============================================================
# _generate_coverage_table
# ============================================================

class TestGenerateCoverageTable:
    def test_retorna_lista_no_vacia(self):
        result = _generate_coverage_table(None, None, None)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_incluye_header(self):
        result = _generate_coverage_table(None, None, None)
        assert any('### Cobertura de Datos' in line for line in result)

    def test_incluye_tabla_markdown(self):
        result = _generate_coverage_table(None, None, None)
        assert any('| Fuente |' in line for line in result)

    def test_sin_pcr_muestra_sin_datos(self):
        result = _generate_coverage_table(None, None, None)
        text = ''.join(result)
        assert 'Sin datos' in text

    def test_sector_results_con_ranking(self):
        sector_results = {
            'ranking': [
                ('XLK', 'Technology', 0.5, 'RANGE'),
                ('XLF', 'Financials', 0.3, 'RANGE'),
                (None, None, None, None),
            ]
        }
        result = _generate_coverage_table(None, None, sector_results)
        text = ''.join(result)
        assert '2/11' in text

    def test_sector_results_ranking_vacio(self):
        sector_results = {'ranking': []}
        result = _generate_coverage_table(None, None, sector_results)
        text = ''.join(result)
        assert '0/11' in text

    def test_pcr_con_fecha_reciente(self):
        pcr_data = {'last_date': pd.Timestamp.now() - pd.Timedelta(days=2)}
        result = _generate_coverage_table(pcr_data, None, None)
        text = ''.join(result)
        # Acepta 2 o 3 por redondeo de dias
        assert ('2 dias' in text) or ('3 dias' in text)

    def test_darkpool_con_semana(self):
        dp_data = {'week': pd.Timestamp.now() - pd.Timedelta(days=20)}
        result = _generate_coverage_table(None, dp_data, None)
        text = ''.join(result)
        assert 'Dark Pool (FINRA)' in text
