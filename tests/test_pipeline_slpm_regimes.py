"""Tests del pipeline: SLPM (fase 8b) + regimes (fase 2).

Cobertura de orquestadores. Los modulos subyacentes (indicators.slpm_v12,
regimes.financial_conditions, etc.) ya estan cubiertos por sus propios tests.
Aqui se verifica el contrato del orquestador: keys de retorno, manejo de
excepciones, rutas de degradacion.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import slpm as pipeline_slpm
from src.pipeline import regimes as pipeline_regimes


# =============================================================================
# src/pipeline/slpm.py
# =============================================================================

def _fake_slpm_result():
    return {
        'sector': 'Technology',
        'sector_etf': 'XLK',
        'state': 'CONFIRMED',
        'leader_integrity': {'lis': 0.48},
        'leader_breadth_v2': {'composite': 0.81},
        'tactical_score': 0.30,
        'structural_score': 0.11,
        'validation_errors': [],
    }


def test_slpm_pasa_resultado_a_caller():
    with patch('indicators.slpm_v12.evaluate_slpm_v12',
               return_value=_fake_slpm_result()):
        out = pipeline_slpm.compute_slpm_v12(
            df_market=pd.DataFrame(),
            sector_results={},
            leader_metrics_for_slpm=[],
            top_sector_flow=None,
            tactical_scores={},
            structural_scores={},
            sector_persistence=None,
        )
        assert out['state'] == 'CONFIRMED'
        assert out['sector_etf'] == 'XLK'


def test_slpm_devuelve_none_si_evaluate_lanza():
    with patch('indicators.slpm_v12.evaluate_slpm_v12',
               side_effect=RuntimeError("boom")):
        out = pipeline_slpm.compute_slpm_v12(
            df_market=pd.DataFrame(),
            sector_results={},
            leader_metrics_for_slpm=[],
            top_sector_flow=None,
            tactical_scores={},
            structural_scores={},
            sector_persistence=None,
        )
        assert out is None


def test_slpm_devuelve_none_si_evaluate_devuelve_none():
    with patch('indicators.slpm_v12.evaluate_slpm_v12', return_value=None):
        out = pipeline_slpm.compute_slpm_v12(
            df_market=pd.DataFrame(),
            sector_results={},
            leader_metrics_for_slpm=[],
            top_sector_flow=None,
            tactical_scores={},
            structural_scores={},
            sector_persistence=None,
        )
        assert out is None


def test_slpm_acepta_validation_errors_no_vacio():
    bad = _fake_slpm_result()
    bad['validation_errors'] = ['e1', 'e2']
    with patch('indicators.slpm_v12.evaluate_slpm_v12', return_value=bad):
        out = pipeline_slpm.compute_slpm_v12(
            df_market=pd.DataFrame(),
            sector_results={},
            leader_metrics_for_slpm=[],
            top_sector_flow=None,
            tactical_scores={},
            structural_scores={},
            sector_persistence=None,
        )
        assert out['validation_errors'] == ['e1', 'e2']


# =============================================================================
# src/pipeline/regimes.py
# =============================================================================

REGIMES_KEYS = {
    'financial_score', 'financial_regime', 'liq_conf',
    'real_liq_score', 'real_liq_regime', 'real_liq_conf', 'real_liq_prev',
    'vol_score', 'vol_regime', 'vol_conf',
    'macro_score', 'macro_regime', 'macro_conf', 'all_signals',
}


def _df_market_with_vix():
    """DataFrame minimo con MultiIndex columnar ('Close','^VIX')."""
    idx = pd.date_range('2026-09-20', periods=5, freq='D')
    cols = pd.MultiIndex.from_tuples([('Close', '^VIX')], names=['field', 'ticker'])
    return pd.DataFrame([15.0, 16.0, 15.5, 17.0, 16.5], index=idx, columns=cols)


def test_regimes_devuelve_14_keys():
    with patch('src.pipeline.regimes.compute_financial_conditions',
               return_value=(0.5, 'ABUNDANTE', 0.9)), \
         patch('src.pipeline.regimes.compute_real_liquidity',
               return_value=(0.2, 'NEUTRAL', 0.8, 0.1)), \
         patch('src.pipeline.regimes.compute_volatility_regime',
               return_value=(0.1, 'NORMAL', 0.9)), \
         patch('src.pipeline.regimes.compute_macro_regime',
               return_value=(0.0, 'MIXED', 0.5, {})), \
         patch('src.pipeline.regimes.get_col',
               return_value=pd.Series([15.0, 16.0, 15.5, 17.0, 16.5])):
        out = pipeline_regimes.compute_all_regimes(
            pd.DataFrame(), pd.DataFrame())
    assert set(out.keys()) == REGIMES_KEYS


def test_regimes_degradacion_liq_real_none():
    """Si compute_real_liquidity devuelve (None, ...), los campos se
    rellenan con valores neutros."""
    with patch('src.pipeline.regimes.compute_financial_conditions',
               return_value=(0.5, 'ABUNDANTE', 0.9)), \
         patch('src.pipeline.regimes.compute_real_liquidity',
               return_value=(None, None, None, None)), \
         patch('src.pipeline.regimes.compute_volatility_regime',
               return_value=(0.1, 'NORMAL', 0.9)), \
         patch('src.pipeline.regimes.compute_macro_regime',
               return_value=(0.0, 'MIXED', 0.5, {})), \
         patch('src.pipeline.regimes.get_col',
               return_value=pd.Series([15.0, 16.0])):
        out = pipeline_regimes.compute_all_regimes(pd.DataFrame(), pd.DataFrame())
    assert out['real_liq_score'] is None
    assert out['real_liq_regime'] == 'N/A'
    assert out['real_liq_conf'] == 0.0


def test_regimes_sin_vix_usa_volatilidad_plana():
    """Si ^VIX no esta en df_market, get_col lanza KeyError -> Series vacia."""
    with patch('src.pipeline.regimes.compute_financial_conditions',
               return_value=(0.5, 'ABUNDANTE', 0.9)), \
         patch('src.pipeline.regimes.compute_real_liquidity',
               return_value=(0.2, 'NEUTRAL', 0.8, 0.1)), \
         patch('src.pipeline.regimes.compute_volatility_regime',
               return_value=(0.0, 'NORMAL', 0.0)) as mock_vol, \
         patch('src.pipeline.regimes.compute_macro_regime',
               return_value=(0.0, 'MIXED', 0.5, {})), \
         patch('src.pipeline.regimes.get_col',
               side_effect=KeyError('^VIX')):
        pipeline_regimes.compute_all_regimes(pd.DataFrame(), pd.DataFrame())
    # Verificar que compute_volatility_regime recibio una Series vacia
    args, _ = mock_vol.call_args
    assert isinstance(args[0], pd.Series)
    assert len(args[0]) == 0


def test_regimes_propaga_temporal_meta_a_macro():
    with patch('src.pipeline.regimes.compute_financial_conditions',
               return_value=(0.5, 'ABUNDANTE', 0.9)), \
         patch('src.pipeline.regimes.compute_real_liquidity',
               return_value=(0.2, 'NEUTRAL', 0.8, 0.1)), \
         patch('src.pipeline.regimes.compute_volatility_regime',
               return_value=(0.1, 'NORMAL', 0.9)), \
         patch('src.pipeline.regimes.compute_macro_regime',
               return_value=(0.0, 'MIXED', 0.5, {})) as mock_macro, \
         patch('src.pipeline.regimes.get_col',
               return_value=pd.Series([15.0, 16.0])):
        tm = {'by_contract': {'EQUITY_EOD': {'coverage': 0.95}}}
        pipeline_regimes.compute_all_regimes(pd.DataFrame(), pd.DataFrame(),
                                             temporal_meta=tm)
    _, kwargs = mock_macro.call_args
    assert kwargs.get('temporal_meta') == tm