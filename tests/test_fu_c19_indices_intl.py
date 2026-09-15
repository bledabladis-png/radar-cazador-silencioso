# -*- coding: utf-8 -*-
"""FU-C19: compute_indices_intl propaga reference_date y run_id.

Origen: Ciclo 2 del subinforme de consumidores de df_market.
Hallazgo C19: indices_intl.py:23 llamaba a download_stock_prices()
sin argumentos, generando una segunda descarga con reference_date
interna distinta de la resuelta en run.py main().

Fix: propagar reference_date y run_id desde run.py a traves de
compute_indices_intl hasta download_stock_prices.
"""

import inspect
from unittest.mock import patch

import src.pipeline.indices_intl as ii_mod


def test_c19_propagates_reference_date_and_run_id():
    """Comportamiento: compute_indices_intl pasa reference_date y run_id."""
    captured = {}

    def fake_download(reference_date=None, run_id=None):
        captured['reference_date'] = reference_date
        captured['run_id'] = run_id
        return None

    def fake_phases(df):
        return ({'^FTSE': 'ACCUMULATION'}, None)

    def fake_leaders(a, b, c):
        return {}

    with patch.object(ii_mod, 'download_stock_prices', fake_download), \
         patch.object(ii_mod, 'compute_index_phases', fake_phases), \
         patch.object(ii_mod, 'select_index_leaders', fake_leaders):
        ii_mod.compute_indices_intl(
            None,
            reference_date='2026-09-15',
            run_id='20260915_120000',
        )

    assert captured['reference_date'] == '2026-09-15'
    assert captured['run_id'] == '20260915_120000'


def test_c19_signature_accepts_kwargs():
    """Regresion estructural: la firma expone reference_date y run_id."""
    sig = inspect.signature(ii_mod.compute_indices_intl)
    params = sig.parameters
    assert 'reference_date' in params
    assert 'run_id' in params
    assert params['reference_date'].default is None
    assert params['run_id'].default is None


def test_c19_source_contains_propagation():
    """Regresion estructural: el fuente propaga los argumentos."""
    src = inspect.getsource(ii_mod)
    assert 'download_stock_prices(reference_date=reference_date, run_id=run_id)' in src