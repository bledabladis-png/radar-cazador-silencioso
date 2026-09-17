# -*- coding: utf-8 -*-
"""Test de regresion I3 (2026-09-18).

El render 'Rendimiento QQQ (Yahoo Finance)' mostraba as_of_date
(timestamp del run, con hora) en vez de effectiveDate (fecha del
dataset). Los valores correctos son:

  effectiveDate: '2026-09-17'         (fecha del dataset)
  as_of_date:    '2026-09-17 23:39:10' (timestamp del run)
"""
import pandas as pd
from src.report.flows_international import render_rendimiento_qqq


def _df(effective=None, as_of=None):
    row = {
        'ytd': 10.0, 'y1': 20.0, 'y3': 90.0, 'y5': 90.0, 'y10': 500.0,
        'inception': 1500.0, 'label': 'marketPrice',
        'displayLabel': 'QQQ (Yahoo Finance)',
    }
    if effective is not None:
        row['effectiveDate'] = effective
    if as_of is not None:
        row['as_of_date'] = as_of
    return pd.DataFrame([row])


def test_muestra_effective_date():
    """Caso normal: effectiveDate presente -> se muestra."""
    df = _df(effective='2026-09-17', as_of='2026-09-17 23:39:10')
    out = render_rendimiento_qqq(df)
    joined = ''.join(out)
    assert 'effectiveDate' in joined
    assert '2026-09-17' in joined
    # No debe mostrar el timestamp del run
    assert '23:39:10' not in joined


def test_fallback_a_as_of_date():
    """Sin effectiveDate: cae a as_of_date pero truncado a fecha."""
    df = _df(as_of='2026-09-17 23:39:10')
    out = render_rendimiento_qqq(df)
    joined = ''.join(out)
    assert 'as_of_date' in joined
    assert '2026-09-17' in joined
    # No debe mostrar la hora
    assert '23:39:10' not in joined


def test_sin_fechas_no_rompe():
    """Sin ninguna fecha: no muestra linea de fecha, no falla."""
    df = _df()
    out = render_rendimiento_qqq(df)
    joined = ''.join(out)
    assert 'Fecha del dato' not in joined
    assert 'Rendimiento QQQ' in joined


def test_effective_date_tiene_prioridad():
    """Si ambas existen: gana effectiveDate."""
    df = _df(effective='2026-09-15', as_of='2026-09-17 23:39:10')
    out = render_rendimiento_qqq(df)
    joined = ''.join(out)
    assert '2026-09-15' in joined
    assert '2026-09-17' not in joined