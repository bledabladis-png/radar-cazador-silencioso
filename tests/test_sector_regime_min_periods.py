# -*- coding: utf-8 -*-
"""F3-05-quinquies (2026-09-30): regresion del min_periods relajado
en compute_sector_scores.

Bug: rs_ret usa pct_change(fill_method=None), que genera 2-3 NaN
consecutivos por festivo NYSE (dia + siguiente por pct_change).
Con min_periods=window (default), rolling(20/50/126) queda NaN
durante 20/50/126 sesiones tras cada festivo. Como rs_mom_20/50/126
son 3 de los 6 componentes del score (peso total 0.50), el score
cae a -0.0 por dispersion sobre solo 3 componentes.

Fix: min_periods relajados (15/40/100), alineados con
utils.robust_zscore.

Verificado en produccion 2026-09-30 sobre data/market_data.parquet:
- Antes: XLB, XLP, XLRE, XLU, XLY en -0.0.
- Despues: solo XLC en -0.0 (legitimo, dispersion real 3.00 > 2).
"""
import inspect

import numpy as np
import pandas as pd

from regimes import sector_regime


def test_compute_sector_scores_usa_min_periods_relajado():
    """Regresion: los 3 rolling del score sectorial deben usar
    min_periods relajado (15/40/100) para tolerar gaps de festivos.

    Sin esto, rs_mom_20/50/126 son NaN 20/50/126 sesiones tras
    cada festivo NYSE, y el score cae a -0.0 en 5-6 sectores.
    """
    src = inspect.getsource(sector_regime.compute_sector_scores)
    assert 'rolling(20, min_periods=15)' in src, (
        'Falta min_periods=15 en rolling(20). Ver F3-05-quinquies.'
    )
    assert 'rolling(50, min_periods=40)' in src, (
        'Falta min_periods=40 en rolling(50). Ver F3-05-quinquies.'
    )
    assert 'rolling(126, min_periods=100)' in src, (
        'Falta min_periods=100 en rolling(126). Ver F3-05-quinquies.'
    )


def test_min_periods_tolera_tres_nan_consecutivos():
    """Documentacion del mecanismo del fix: rolling(20, min_periods=15)
    tolera 3 NaN consecutivos en una ventana de 20; rolling(20) default
    no. Esto es lo que evita el colapso del score a -0.0 tras un
    festivo NYSE.
    """
    n = 100
    rng = np.random.RandomState(42)
    s = pd.Series(rng.normal(0, 0.01, n))
    # 3 NaN consecutivos a -10 (patron festivo + pct_change)
    s.iloc[-10] = np.nan
    s.iloc[-9] = np.nan
    s.iloc[-8] = np.nan

    # Sin min_periods relajado -> NaN
    assert pd.isna(s.rolling(20).mean().iloc[-1])
    assert pd.isna(s.rolling(20).std().iloc[-1])

    # Con min_periods=15 -> valido
    assert not pd.isna(s.rolling(20, min_periods=15).mean().iloc[-1])
    assert not pd.isna(s.rolling(20, min_periods=15).std().iloc[-1])
