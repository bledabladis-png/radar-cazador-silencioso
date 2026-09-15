# -*- coding: utf-8 -*-
"""FU-C16: compute_sector_metrics propaga effective_meta (FU-020).

Origen: Ciclo 2 del subinforme de consumidores de df_market.
Hallazgo C16: sector_metrics.py usaba df_stocks.index[-1] como
reference_date para los writers de sector_concentration y
leader_representativeness, sin declarar ni consumir el effective_meta
de FU-020. R3 (FU-020) exige que la resolucion temporal se comparta
entre metricas derivadas del mismo universo.

Fix: compute_sector_metrics acepta effective_meta=None. Cuando esta
presente, su 'date' se usa como reference_date. Fallback a
df_stocks.index[-1] (comportamiento historico) cuando es None.
"""

import inspect
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import src.pipeline.sector_metrics as sm_mod


def _minimal_df_stocks():
    return pd.DataFrame(
        {'Close': [100.0, 101.0, 102.0]},
        index=pd.date_range('2026-09-10', periods=3, freq='D'),
    )


def test_c16_uses_effective_meta_date_when_provided():
    captured = {}

    def fake_conc(df_stocks, holdings_df, leader_df, full_metrics_df, reference_date=None, sc_path=None):
        captured['concentration_ref'] = reference_date
        return pd.DataFrame()

    def fake_repr(leader_df, reference_date=None):
        captured['repr_ref'] = reference_date
        return pd.DataFrame()

    meta = {'date': pd.Timestamp('2026-09-14'), 'coverage': 0.98}

    with patch.object(sm_mod, '_compute_concentration', fake_conc), \
         patch.object(sm_mod, '_compute_representativeness', fake_repr), \
         patch.object(sm_mod, '_compute_divergencia', lambda *a, **k: None), \
         patch.object(sm_mod, '_compute_wyckoff', lambda *a, **k: None), \
         patch.object(sm_mod, '_compute_rs_internal', lambda *a, **k: None):
        sm_mod.compute_sector_metrics(
            _minimal_df_stocks(), None, None, None, None,
            effective_meta=meta,
        )

    assert captured['concentration_ref'] == pd.Timestamp('2026-09-14')
    assert captured['repr_ref'] == pd.Timestamp('2026-09-14')


def test_c16_falls_back_to_index_last_when_meta_absent():
    captured = {}

    def fake_conc(df_stocks, holdings_df, leader_df, full_metrics_df, reference_date=None, sc_path=None):
        captured['concentration_ref'] = reference_date
        return pd.DataFrame()

    def fake_repr(leader_df, reference_date=None):
        captured['repr_ref'] = reference_date
        return pd.DataFrame()

    df_stocks = _minimal_df_stocks()
    expected = df_stocks.index[-1]

    with patch.object(sm_mod, '_compute_concentration', fake_conc), \
         patch.object(sm_mod, '_compute_representativeness', fake_repr), \
         patch.object(sm_mod, '_compute_divergencia', lambda *a, **k: None), \
         patch.object(sm_mod, '_compute_wyckoff', lambda *a, **k: None), \
         patch.object(sm_mod, '_compute_rs_internal', lambda *a, **k: None):
        sm_mod.compute_sector_metrics(
            df_stocks, None, None, None, None,
            effective_meta=None,
        )

    assert captured['concentration_ref'] == expected
    assert captured['repr_ref'] == expected


def test_c16_signature_accepts_effective_meta():
    sig = inspect.signature(sm_mod.compute_sector_metrics)
    params = sig.parameters
    assert 'effective_meta' in params
    assert params['effective_meta'].default is None


def test_c16_run_py_propagates_effective_meta():
    src = Path('run.py').read_text(encoding='utf-8-sig')
    assert 'compute_sector_metrics(df_stocks, holdings_df, leader_df, full_metrics_df, df_market, effective_meta=df_stocks_effective_meta)' in src