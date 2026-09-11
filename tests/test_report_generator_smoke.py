"""
Smoke test de generate_daily_report.

Verifica que el reporte se genera sin crashear con una fixture minima
de los 14 parametros obligatorios. NO valida contenido especifico
(depende de datos reales). Solo comprueba:
- No crash.
- Archivo generado.
- Secciones clave presentes.
- No hay errores propagados en el output.
"""
import pandas as pd
import numpy as np

from src.report_generator import generate_daily_report


def _make_minimal_fixture():
    """Construye los 14 parametros obligatorios con datos sinteticos."""
    idx = pd.date_range('2026-01-01', periods=10, freq='D')

    macro_score = pd.Series([0.1, 0.05, 0.0, -0.05, 0.02, 0.08, 0.1, 0.05, 0.0, 0.02], index=idx)
    liquidity_score = pd.Series([-0.2]*10, index=idx)
    volatility_score = pd.Series([-0.5]*10, index=idx)

    sector_results = {
        'regime': 'NARROW RALLY',
        'ranking': [
            ('XLK', 'Technology', 0.61, 'RANGE'),
            ('XLE', 'Energy', 0.19, 'RANGE'),
            ('XLF', 'Financials', -0.00, 'RANGE'),
            ('XLV', 'Healthcare', -0.00, 'RANGE'),
            ('XLP', 'Consumer Staples', -0.00, 'RANGE'),
            ('XLI', 'Industrials', -0.00, 'RANGE'),
            ('XLY', 'Consumer Discretionary', -0.02, 'RANGE'),
            ('XLB', 'Materials', -0.26, 'RANGE'),
            ('XLU', 'Utilities', -0.00, 'RANGE'),
            ('XLRE', 'Real Estate', -0.00, 'RANGE'),
            ('XLC', 'Communication Services', -0.00, 'RANGE'),
        ],
    }

    sector_price_rank = [('XLK', 0.5), ('XLE', 0.3), ('XLF', 0.1)]
    sector_flow_rank = [('XLK', 0.2), ('XLE', 0.1), ('XLF', -0.1)]
    otros_price_rank = [('GLD', 0.1), ('TLT', -0.1)]
    otros_flow_rank = [('GLD', 0.05), ('TLT', -0.05)]

    return (
        macro_score, 'MIXED', 0.50,
        liquidity_score, 'ESTRECHA', 0.58,
        volatility_score, 'LOW', 0.38,
        sector_results,
        sector_price_rank, sector_flow_rank,
        otros_price_rank, otros_flow_rank,
    )


def test_generate_daily_report_smoke(tmp_path):
    """Smoke test: la funcion no debe crashear con fixture minima."""
    fixture = _make_minimal_fixture()
    output = tmp_path / 'reporte_test.md'

    # Debe ejecutarse sin lanzar excepcion
    generate_daily_report(*fixture, output_path=str(output))

    # Verificar que el archivo se creo
    assert output.exists(), 'El reporte no se genero en disco'

    content = output.read_text(encoding='utf-8')

    # Tamanio razonable
    assert len(content) > 100, f'Reporte demasiado corto: {len(content)} bytes'

    # Header presente
    assert 'MACRO SECTORIAL' in content, 'Falta header del reporte'

    # Secciones incondicionales (siempre presentes independientemente
    # de los datos opcionales pasados).
    secciones_obligatorias = [
        '## Resumen de Regimenes',
        '## Momentum de Precio - Sectores',
        '## Rankings Sectoriales',
        '## Estado Actual',
    ]
    for sec in secciones_obligatorias:
        assert sec in content, f'Falta seccion obligatoria: {sec}'

    # No debe contener tracebacks propagados
    assert 'Traceback' not in content, 'Traceback propagado al reporte'
    assert 'NameError' not in content, 'NameError en el reporte'
    assert 'AttributeError' not in content, 'AttributeError en el reporte'


def test_generate_daily_report_macro_score_nan(tmp_path):
    """Smoke test con macro_score NaN (caso edge del reporte)."""
    fixture = list(_make_minimal_fixture())
    # Sustituir macro_score por serie de NaN
    idx = pd.date_range('2026-01-01', periods=10, freq='D')
    fixture[0] = pd.Series([np.nan]*10, index=idx)

    output = tmp_path / 'reporte_nan.md'
    generate_daily_report(*fixture, output_path=str(output))

    assert output.exists()
    content = output.read_text(encoding='utf-8')
    # Debe mostrar N/D en vez de crashear
    assert 'N/D' in content
    assert 'Traceback' not in content


def test_generate_daily_report_with_breadth_values(tmp_path):
    """Smoke test con breadth_values presente (seccion condicional)."""
    fixture = list(_make_minimal_fixture())
    breadth_values = {
        '% sobre EMA20': 0.18,
        '% sobre EMA50': 0.36,
        '% sobre EMA200': 0.73,
        'New Highs (%)': 0.0,
        'New Lows (%)': 0.0,
        'EMA20 count': 2,
        'EMA50 count': 4,
        'EMA200 count': 8,
        'New Highs count': 0,
        'New Lows count': 0,
    }
    output = tmp_path / 'reporte_breadth.md'
    generate_daily_report(*fixture, breadth_values=breadth_values, output_path=str(output))

    assert output.exists()
    content = output.read_text(encoding='utf-8')
    assert '## Breadth de Mercado' in content, 'Falta seccion Breadth de Mercado con datos'
    assert 'Traceback' not in content
