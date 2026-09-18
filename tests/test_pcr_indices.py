# -*- coding: utf-8 -*-
"""Test de regresion PCR Indices (2026-09-18).

Bug: compute_pcr_signals() no incluia 'index_pcr' en el dict de
retorno, aunque data['index_pcr'] se usaba internamente (L65) y se
guardaba en el CSV historico (L97). sentiment.py:34 lo leia y caia
al default np.nan -> N/D en el reporte.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators.options import compute_pcr_signals


def _fake_cboe_data():
    return {
        'date': '2026-09-17',
        'total_pcr': 0.79,
        'index_pcr': 1.07,
        'equity_pcr': 0.52,
        'etp_pcr': 0.76,
        'vix_pcr': 0.50,
        'spx_pcr': 1.15,
        'total_call_volume': 7656055,
        'total_put_volume': 6000000,
        'total_volume': 13656055,
        'total_call_oi': 5000000,
        'total_put_oi': 3800000,
        'total_oi': 8800000,
        'index_call_volume': 2000000,
        'index_put_volume': 2140000,
        'index_volume': 4140000,
        'index_call_oi': 1000000,
        'index_put_oi': 1070000,
        'index_oi': 2070000,
        'equity_call_volume': 3000000,
        'equity_put_volume': 1560000,
        'equity_volume': 4560000,
        'equity_call_oi': 2000000,
        'equity_put_oi': 1040000,
        'equity_oi': 3040000,
    }


def test_return_dict_incluye_index_pcr():
    """El dict de retorno debe exponer 'index_pcr'."""
    with patch('indicators.options.CboeProvider') as MockProvider:
        instance = MockProvider.return_value
        instance.is_available.return_value = True
        instance.get_options_data.return_value = _fake_cboe_data()

        result = compute_pcr_signals()

    assert result is not None
    assert 'index_pcr' in result, "Bug: 'index_pcr' falta en el dict de retorno"
    assert result['index_pcr'] == pytest.approx(1.07)


def test_index_pcr_coincide_con_equity_y_otros():
    """Sanity: los 5 PCRs deben estar presentes."""
    with patch('indicators.options.CboeProvider') as MockProvider:
        instance = MockProvider.return_value
        instance.is_available.return_value = True
        instance.get_options_data.return_value = _fake_cboe_data()

        result = compute_pcr_signals()

    for key in ('total_pcr', 'index_pcr', 'equity_pcr', 'etp_pcr',
                'vix_pcr', 'spx_pcr'):
        assert key in result, f"Falta clave {key}"
        assert result[key] is not None