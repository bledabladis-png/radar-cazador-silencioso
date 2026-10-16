# -*- coding: utf-8 -*-
"""Test de contrato: resolve_target_session interpreta input como UTC.

Cubre la regresion detectada el 2026-10-16: la funcion documentaba
"default: UTC" pero pasaba naive a last_expected_market_date, que lo
interpreta como Madrid. Ventana de divergencia detectada: 21:00-22:00
UTC (2h/dia). Con el fix, naive y aware dan identico resultado en
las 24 horas del dia.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.pipeline_gate import resolve_target_session


# Viernes habil
VIERNES = datetime(2026, 10, 16, 0, 0)
# Sabado
SABADO = datetime(2026, 10, 17, 0, 0)


def _naive(base, h):
    return base.replace(hour=h, minute=0)


def _aware(base, h):
    return base.replace(hour=h, minute=0, tzinfo=timezone.utc)


def test_24_horas_naive_igual_aware_viernes():
    """En las 24 horas de un viernes, naive y aware dan lo mismo."""
    for h in range(24):
        t_naive = resolve_target_session(_naive(VIERNES, h))
        t_aware = resolve_target_session(_aware(VIERNES, h))
        assert t_naive == t_aware, (
            f'hora {h:02d} UTC: naive={t_naive} != aware={t_aware}'
        )


def test_24_horas_naive_igual_aware_sabado():
    """En las 24 horas de un sabado, naive y aware dan lo mismo."""
    for h in range(24):
        t_naive = resolve_target_session(_naive(SABADO, h))
        t_aware = resolve_target_session(_aware(SABADO, h))
        assert t_naive == t_aware, (
            f'hora {h:02d} UTC sabado: naive={t_naive} != aware={t_aware}'
        )


def test_ventana_21_22_utc_viernes_devuelve_16_oct():
    """Regresion: 21:00 y 22:00 UTC viernes deben devolver 16-oct
    (la sesion del dia, ya disponible). Antes devolvian 15-oct por
    interpretar naive como Madrid."""
    assert resolve_target_session(_naive(VIERNES, 21)) == '2026-10-16'
    assert resolve_target_session(_naive(VIERNES, 22)) == '2026-10-16'
    # Con aware, mismo resultado
    assert resolve_target_session(_aware(VIERNES, 21)) == '2026-10-16'
    assert resolve_target_session(_aware(VIERNES, 22)) == '2026-10-16'


def test_docstring_afirma_contrato():
    """El docstring debe declarar el contrato UTC explicitamente."""
    doc = resolve_target_session.__doc__ or ''
    assert 'UTC-aware' in doc, 'docstring debe mencionar UTC-aware'
    assert 'naive' in doc.lower(), 'docstring debe mencionar naive'


def test_slots_del_cron_siguen_siendo_correctos():
    """Los 5 slots del cron siguen resolviendo como antes."""
    casos = [
        # (dia base, hora, minuto, target esperado)
        (datetime(2026, 9, 25), 23, 17, '2026-09-25'),  # viernes 23:17 UTC
        (datetime(2026, 9, 25), 3, 17, '2026-09-24'),   # viernes 03:17 UTC
        (datetime(2026, 9, 25), 5, 17, '2026-09-24'),
        (datetime(2026, 9, 25), 7, 17, '2026-09-24'),
        (datetime(2026, 9, 25), 11, 17, '2026-09-24'),
    ]
    for base, h, m, esperado in casos:
        now = base.replace(hour=h, minute=m)
        assert resolve_target_session(now) == esperado, (
            f'{base.date()} {h:02d}:{m:02d} UTC -> {esperado}'
        )


def test_default_sin_argumento_no_falla():
    """Sin argumento, la funcion resuelve ahora UTC-aware."""
    out = resolve_target_session()
    # No podemos predecir el valor exacto, pero debe ser ISO date valido
    assert isinstance(out, str)
    assert len(out) == 10
    assert out[4] == '-' and out[7] == '-'