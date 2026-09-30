# -*- coding: utf-8 -*-
"""D1 (2026-09-30): contrato de la dispersion en sector_regime.

Verifica las propiedades de _dispersion introducidas tras la auditoria D1.7:
- componentes acotados en [-1, +1] -> dispersion en [0, 1];
- dispersion = std poblacional (ddof=0), NO dividida por |mean|;
- multiplier = clip(1 - 0.5 * dispersion, 0, 1) en [0.5, 1].

Auditoria externa (2026-09-30): sin ddof=0 explicito, std([-1, +1]) = sqrt(2)
en pandas default, y el multiplier cae a 0.293 en vez de 0.5. Los 4 tests
siguientes bloquean esa regresion.
"""
import numpy as np
import pandas as pd

from regimes import sector_regime


PENALTY = 0.5  # SECTOR_DISPERSION_PENALTY


def _multiplier(dispersion):
    return max(0.0, min(1.0, 1 - PENALTY * dispersion))


def test_dispersion_todos_iguales_multiplier_uno():
    """Componentes identicos -> dispersion 0 -> multiplier 1."""
    row = pd.Series([0.5, 0.5, 0.5, 0.5])
    d = sector_regime._dispersion(row)
    assert d == 0.0, "dispersion={}".format(d)
    assert _multiplier(d) == 1.0


def test_dispersion_maxima_multiplier_medio():
    """Componentes [-1, +1] -> dispersion poblacional 1 -> multiplier 0.5.

    Sin ddof=0, pandas devuelve sqrt(2) y el multiplier cae a 0.293.
    Este test bloquea esa regresion.
    """
    row = pd.Series([-1.0, 1.0])
    d = sector_regime._dispersion(row)
    assert abs(d - 1.0) < 1e-9, "dispersion={} (esperado 1.0)".format(d)
    m = _multiplier(d)
    assert abs(m - 0.5) < 1e-9, "multiplier={} (esperado 0.5)".format(m)


def test_dispersion_media_cero_no_explota():
    """Componentes que se cancelan (media ~ 0) -> dispersion finita."""
    row = pd.Series([0.8, -0.8, 0.1, -0.1])
    d = sector_regime._dispersion(row)
    assert np.isfinite(d), "dispersion no finita: {}".format(d)
    assert 0.0 <= d <= 1.0


def test_dispersion_rango_valido_y_multiplier_acotado():
    """Propiedad: para todo vector en [-1, +1], dispersion en [0, 1] y
    multiplier en [0.5, 1]. 50 muestras aleatorias."""
    rng = np.random.RandomState(42)
    for _ in range(50):
        n = rng.randint(2, 10)
        vals = rng.uniform(-1, 1, n)
        row = pd.Series(vals)
        d = sector_regime._dispersion(row)
        assert 0.0 <= d <= 1.0, "dispersion fuera de rango: {}".format(d)
        m = _multiplier(d)
        assert 0.5 <= m <= 1.0, "multiplier fuera de rango: {}".format(m)


def test_dispersion_menos_de_dos_valores():
    """Con <2 valores no-NaN, dispersion 0."""
    row = pd.Series([0.5])
    assert sector_regime._dispersion(row) == 0.0
    row = pd.Series([np.nan, 0.5, np.nan])
    assert sector_regime._dispersion(row) == 0.0
