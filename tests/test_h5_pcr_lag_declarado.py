# -*- coding: utf-8 -*-
"""Centinela H5 (auditoria 2026-11-07): lag del PCR respecto al VIX.

Bug real: la tabla 'Estructura de volatilidad' publicaba una sola
`Fecha` (la del VIX) para toda la fila, incluyendo el PCR. Si el
PCR venia del dia anterior (CBOE no habia publicado aun), el
lector veia "PCR z 0.20" bajo fecha 5/11 cuando el PCR era del 4/11.

Causa raiz: el dato `last_date` existia en `pcr_data`
(indicators/options.py:237), se perdia en `market_data.py`
(solo copiaba zscore y percentile_20d), y el render
(src/report/volatility_mte.py) no lo declaraba.

Fix (commit asociado):
- market_data._compute_vol_structure propaga `date` de pcr_data.
- volatility_structure.compute_volatility_structure lee `pcr_date`
  y calcula `pcr_lag_days` respecto a `_obs_date` (VIX).
- render_estructura_volatilidad anade '(lag Nd)' a PCR z y
  Perc PCR, y una nota al pie con la fecha del PCR.

Sin hardcode de fuentes: cualquier PCR con fecha distinta al VIX
se declara.
"""
from __future__ import annotations

import pandas as pd
import pytest

from indicators.volatility_structure import compute_volatility_structure
from src.report.volatility_mte import render_estructura_volatilidad


def _make_market_df(n=100):
    idx = pd.date_range('2026-01-01', periods=n, freq='D')
    return pd.DataFrame({
        '^VIX': [20.0] * n,
        '^VIX3M': [22.0] * n,
    }, index=idx)


class TestComputeVolatilityStructurePcrLag:

    def test_sin_pcr_pcr_date_es_none_y_lag_nan(self):
        out = compute_volatility_structure(_make_market_df())
        assert 'pcr_date' in out.columns
        assert 'pcr_lag_days' in out.columns
        assert out['pcr_date'].iloc[0] is None or pd.isna(out['pcr_date'].iloc[0])
        assert pd.isna(out['pcr_lag_days'].iloc[0])

    def test_pcr_date_igual_a_vix_lag_cero(self):
        df = _make_market_df()
        vix_date = df.index[-1].strftime('%Y-%m-%d')
        pcr = {'zscore': 0.5, 'percentile_20d': 50.0, 'date': vix_date}
        out = compute_volatility_structure(df, pcr_data=pcr)
        assert out['pcr_date'].iloc[0] == vix_date
        assert int(out['pcr_lag_days'].iloc[0]) == 0

    @pytest.mark.parametrize('lag_dias', [1, 2, 5])
    def test_pcr_date_anterior_lag_correcto(self, lag_dias):
        df = _make_market_df()
        vix_date = df.index[-1]
        pcr_date = (vix_date - pd.Timedelta(days=lag_dias)).strftime('%Y-%m-%d')
        pcr = {'zscore': 0.5, 'percentile_20d': 50.0, 'date': pcr_date}
        out = compute_volatility_structure(df, pcr_data=pcr)
        assert int(out['pcr_lag_days'].iloc[0]) == lag_dias

    def test_pcr_sin_date_lag_nan(self):
        """pcr_data sin 'date' (compat con tests viejos): no rompe."""
        pcr = {'zscore': 0.5, 'percentile_20d': 50.0}
        out = compute_volatility_structure(_make_market_df(), pcr_data=pcr)
        assert pd.isna(out['pcr_lag_days'].iloc[0])


class TestRenderDeclaraLag:

    def _df(self, pcr_lag, pcr_date=None):
        return pd.DataFrame([{
            'date': '2026-11-05',
            'vix_level': 28.18,
            'vix_percentile_20d': 1.0,
            'vix_percentile_60d': 1.0,
            'term_structure_ratio': 0.91,
            'pcr_zscore': 0.20,
            'pcr_percentile_20d': 10.0,
            'pcr_date': pcr_date,
            'pcr_lag_days': pcr_lag,
            'volatility_reading': 'Volatilidad elevada reciente',
            'term_structure_reading': 'Backwardation',
        }])

    def test_sin_lag_no_anade_sufijo(self):
        joined = ''.join(render_estructura_volatilidad(self._df(0)))
        assert 'lag 0d' not in joined
        assert '(lag' not in joined

    @pytest.mark.parametrize('lag', [1, 2, 5])
    def test_con_lag_anade_sufijo_y_nota(self, lag):
        out = render_estructura_volatilidad(
            self._df(lag, pcr_date='2026-11-04'))
        joined = ''.join(out)
        assert f'(lag {lag}d)' in joined
        assert 'PCR (CBOE)' in joined
        assert '2026-11-04' in joined

    def test_lag_nan_se_trata_como_sin_lag(self):
        """pcr_lag_days NaN (pcr_data sin fecha): sin sufijo, sin nota."""
        joined = ''.join(render_estructura_volatilidad(self._df(float('nan'))))
        assert '(lag' not in joined
        assert 'PCR (CBOE)' not in joined


class TestPipelineTransportaFecha:

    def test_compute_vol_structure_propaga_last_date(self):
        """_compute_vol_structure pasa pcr_data['last_date'] al productor."""
        from unittest.mock import patch
        from src.pipeline.market_data import _compute_vol_structure

        fake_df = pd.DataFrame([{'date': '2026-11-05', 'pcr_lag_days': 1}])
        pcr_data = {
            'z_score': 0.5, 'percentile_20d': 50.0, 'last_date': '2026-11-04',
        }
        captured = {}
        def _fake(dm, pcr_data=None, **kw):
            captured['pcr_data'] = pcr_data
            return fake_df
        with patch(
                'indicators.volatility_structure.compute_volatility_structure',
                side_effect=_fake), \
             patch('src.pipeline.market_data.persist'):
            _compute_vol_structure(pd.DataFrame(), pcr_data)
        assert captured['pcr_data']['date'] == '2026-11-04', (
            'market_data no propaga last_date: ' + repr(captured['pcr_data']))
